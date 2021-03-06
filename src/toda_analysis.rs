use jieba_rs::Jieba;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io;
use std::path;
use std::path::PathBuf;
use word2vec;
use zip;

use ndarray::arr2;
use ndarray::prelude::*;
use ndarray::ArrayView;
use ndarray::ShapeBuilder;
use ndarray::{array, s, Axis};
// use ndarray_rand::rand::SeedableRng;
use ndarray::OwnedRepr;

use num_traits::cast::ToPrimitive;
// use num_traits::cast::ToPrimitive;
use approx::assert_abs_diff_eq;
use linfa::traits::{Fit, FitWith, Predict};
use linfa::DatasetBase;
use linfa_clustering::{generate_blobs, IncrKMeansError, KMeans, KMeansParams};
use linfa_nn;
use rand_core::SeedableRng;
use rand_isaac::Isaac64Rng;

use std::time::Instant;

use crate:: yadan_typing::{DialogueTurn, Role, UnlabeledDialogue};

// Given a path of directory, return the raw corpus. The raw corpus
// can be seen as a vector of UnlabeledDialogue.
pub fn read_data_from_dir(dpath: &str) -> Vec<UnlabeledDialogue> {
    let mut results: Vec<UnlabeledDialogue> = vec![];
    let paths = fs::read_dir(dpath).unwrap();
    for path in paths {
        let mut dialogue_content: UnlabeledDialogue = UnlabeledDialogue::default();
        let p = path.unwrap().path();
        if p.is_file() {
            let data: Vec<String> = fs::read_to_string(p)
                .unwrap()
                .split("\n")
                .map(|x| x.to_string())
                .collect();
            let mut turn = ("".to_owned(), "".to_owned());
            for (i, line) in data.iter().enumerate() {
                if i % 2 == 0 {
                    turn.0 = line.clone();
                    if i == data.len() {
                        turn.1 = "".to_owned();
                        dialogue_content.contents.push(turn.clone());
                    }
                } else {
                    turn.1 = line.clone();
                    dialogue_content.contents.push(turn.clone());
                }
            }
        }
        results.push(dialogue_content.clone());
    }
    results
}

// Given a zip file path, return the raw corpus. The raw corpus
// can be seen as a vector of UnlabeledDialogue.
pub fn read_data_from_zip(fpath: &str) -> Vec<UnlabeledDialogue> {
    let results: Vec<UnlabeledDialogue> = vec![];

    let dst_path = "./unlabeled/";
    let target_dir = unzip(fpath, dst_path);
    let directory_name = fpath.split(".").nth(0).unwrap();
    // println!("{}",directory_name);
    let results = read_data_from_dir(&target_dir);
    results
}

// Given a path (src_path) of zip file and the directory path you want
// to unzip to, execute the unzip operation and return the example directory
// name.
fn unzip(src_path: &str, dst_path: &str) -> String {
    let fname = std::path::Path::new(src_path);
    let file = fs::File::open(&fname).unwrap();

    let mut results = "".to_owned();

    let mut archive = zip::ZipArchive::new(file).unwrap();

    for i in 0..archive.len() {
        let mut file = archive.by_index(i).unwrap();
        // println!("{:?}",&file);
        let outpath = match file.enclosed_name() {
            Some(path) => path.to_owned(),
            None => continue,
        };
        let str_opath = outpath.clone().into_os_string().into_string().unwrap();
        // println!("{:?}",&str_opath);
        // let outpath=std::path::Path::new(dst_path+outpath.unwrap().to_str());
        // break;
        let final_str: String = dst_path.to_owned() + &str_opath;
        let outpath = PathBuf::from(final_str.clone());

        if i == 0 {
            results = final_str.clone();
        }

        {
            let comment = file.comment();
            if !comment.is_empty() {
                println!("File {} comment: {}", i, comment);
            }
        }

        if (*file.name()).ends_with('/') {
            println!("File {} extracted to \"{}\"", i, outpath.display());
            fs::create_dir_all(&outpath).unwrap();
        } else {
            println!(
                "File {} extracted to \"{}\" ({} bytes)",
                i,
                outpath.display(),
                file.size()
            );
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    fs::create_dir_all(&p).unwrap();
                }
            }
            let mut outfile = fs::File::create(&outpath).unwrap();
            io::copy(&mut file, &mut outfile).unwrap();
        }

        // Get and Set permissions
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            if let Some(mode) = file.unix_mode() {
                fs::set_permissions(&outpath, fs::Permissions::from_mode(mode)).unwrap();
            }
        }
    }

    results
}

//--------------------------------------------------
//     blew functions are for chinese segmentation
//--------------------------------------------------

// Specifically designed for Chinese.
// Given a segmentor (e.g. Jieba) and a dialogue turn, return a vector of string tokenizered.
pub fn segement_per_turn<'a>(
    segmentor: &'a Jieba,
    dialogue_turn: &'a DialogueTurn,
) -> Vec<&'a str> {
    let mut results = segmentor.cut(&dialogue_turn.0, false).to_owned();
    results.append(&mut segmentor.cut(&dialogue_turn.1, false));

    results
}

// Specifically designed for Chinese.
// Given a segmentor (e.g. Jieba) and an unlabeled dialogue, return a vector of string tokenizered.
pub fn segement_unlabeled_dialogue<'a>(
    segmentor: &'a Jieba,
    dialogue: &'a UnlabeledDialogue,
) -> Vec<&'a str> {
    let mut results: Vec<&'a str> = vec![];
    for turn in &dialogue.contents {
        results.append(&mut segement_per_turn(&segmentor, &turn));
    }

    results
}

// Given a unlabeled dialogue, split it with space. Sepecifically designed for English.
pub fn split_with_seperator<'a>(dialogue: &'a UnlabeledDialogue) -> Vec<&'a str> {
    let mut results: Vec<&'a str> = vec![];
    for turn in &dialogue.contents {
        let turn_splited1 = turn.0.split(" ");
        let turn_splited2 = turn.1.split(" ");
        results.append(&mut turn_splited1.collect());
        results.append(&mut turn_splited2.collect());
    }

    results
}

//---------------------------------------------------------------
// unsupervised dialogue mining.
// Mining domains, slots and values from unlabeled dialogue corpus.
//---------------------------------------------------------------

pub fn domain_mining(
    dialogues: &Vec<UnlabeledDialogue>,
    wordvec_path: &str,
) -> HashMap<i32, Vec<UnlabeledDialogue>> {
    let topk = 5;
    let vecd = 300;
    let mut domain_clusters: HashMap<i32, Vec<UnlabeledDialogue>> = HashMap::new();
    assert!(dialogues.len() >= 1);

    println!("Begin to load pre-trained word vector.");
    let now = Instant::now();
    let word_vec_model = word2vec::wordvectors::WordVector::load_from_binary(wordvec_path)
        .expect("Unable to load word2vec models.");
    println!(
        "pre-trained word vector loaded done, time:{}",
        now.elapsed().as_secs()
    );

    // make seperatation.
    let is_chinese: bool = is_chinese(&dialogues[0]);
    let mut all_segments: HashMap<i32, Vec<&str>> = HashMap::new();
    let mut dialogue_embedds: Vec<Vec<f64>> = vec![];
    let mut dialogue_lss: Vec<Vec<&str>> = vec![];

    let zero_vecs: Vec<f32> = vec![0.0; vecd];
    if is_chinese {
        let jiebba = Jieba::new();
        for (i, dialogue) in dialogues.iter().enumerate() {
            let segments = segement_unlabeled_dialogue(&jiebba, dialogue);
            all_segments.insert(i as i32, segments.clone());
            dialogue_lss.push(segments);
        }
        let sorted_words = tf_df(&dialogue_lss);
        let mut points = Array::zeros((dialogue_lss.len(), vecd));
        for (i, d) in dialogue_lss.iter().enumerate() {
            let key_words = find_keyword(&d, &sorted_words, topk);
            let mut key_embedds: Vec<f64> = vec![];
            for w in key_words {
                let v: Vec<f32> = word_vec_model.get_vector(w).unwrap_or(&zero_vecs).clone();
                let mut v: Vec<_> = v.iter().map(|&x| x.to_f64().unwrap()).collect();
                key_embedds.append(&mut v);
            }
            let key_embedds: Array2<f64> =
                Array::from_shape_vec((topk, vecd), key_embedds.clone()).unwrap();
            let embedded = max_pooling(&key_embedds);
            let mut p = points.slice_mut(s![i, ..]);
            p = ndarray::ArrayViewMut1::from(embedded.clone().view_mut());
        }

        // now make clustering.
        println!("the points is:\n ------------\n{:?}", &points);
        let model = make_clustering(5, &points, 100, 1.0);
        println!("{:?}", model);
    } else {
        for (i, dialogue) in dialogues.iter().enumerate() {
            let segments = split_with_seperator(dialogue);
            all_segments.insert(i as i32, segments.clone());
            dialogue_lss.push(segments);
        }
        let sorted_words = tf_df(&dialogue_lss);
        let mut points = Array::zeros((dialogue_lss.len(), vecd));
        for (i, d) in dialogue_lss.iter().enumerate() {
            let key_words = find_keyword(&d, &sorted_words, topk);
            let mut key_embedds: Vec<f64> = vec![];
            for w in key_words {
                let v: Vec<f32> = word_vec_model.get_vector(w).unwrap_or(&zero_vecs).clone();
                let mut v: Vec<_> = v.iter().map(|&x| x.to_f64().unwrap()).collect();
                key_embedds.append(&mut v);
            }
            let key_embedds: Array2<f64> =
                Array::from_shape_vec((topk, vecd), key_embedds.clone()).unwrap();
            let embedded = max_pooling(&key_embedds);
            let mut p = points.slice_mut(s![i, ..]);
            p = ndarray::ArrayViewMut1::from(embedded.clone().view_mut());
        }

        // now make clustering.
        println!("the points is:\n ------------\n{:?}", &points);
        let model = make_clustering(5, &points, 100, 1.0);
        println!("{:?}", model);
    }
    domain_clusters
}

pub fn make_clustering(
    k: usize,
    points: &Array2<f64>,
    batch_size: usize,
    tolerance: f64,
) -> linfa_clustering::KMeans<f64, linfa_nn::distance::L2Dist> {
    // Our random number generator, seeded for reproducibility
    let seed = 42;
    let mut rng = Isaac64Rng::seed_from_u64(seed);

    // Shuffling the dataset is one way of ensuring that the batches contain random points from
    // the dataset, which is required for the algorithm to work properly
    let observations = DatasetBase::from(points.clone()).shuffle(&mut rng);
    let clf = KMeans::params_with_rng(k, rng.clone()).tolerance(tolerance);

    // Repeatedly run fit_with on every batch in the dataset until we have converged
    let model = observations
        .sample_chunks(batch_size)
        .cycle()
        .try_fold(None, |current, batch| {
            match clf.fit_with(current, &batch) {
                // Early stop condition for the kmeans loop
                Ok(model) => Err(model),
                // Continue running if not converged
                Err(IncrKMeansError::NotConverged(model)) => Ok(Some(model)),
                Err(err) => panic!("unexpected kmeans error: {}", err),
            }
        })
        .unwrap_err();

    model
}

// pooling a matrix with shape m*n to a vector with shape n by max operation.
fn max_pooling(embedds: &Array2<f64>) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> {
    let results = embedds.map_axis(Axis(0), |view| {
        view.into_iter()
            .max_by(|a, b| b.partial_cmp(a).unwrap())
            .unwrap()
            .clone()
    });
    results
}

// pooling a matrix with shape m*n to a vector with shape n by mean operation.
fn mean_pooling(embedds: &Array2<f64>) -> Array1<f64> {
    let results = embedds.map_axis(Axis(0), |view| {
        (view.iter().sum::<f64>()) / (view.iter().len() as f64)
    });
    results
}

// given tokenized dialogue corpus, return a rank of words depending on their
// importances under this corpus. Here TF_IDF algorithm was used, which calculate
// the text frequency of a word in a vector as tf, and then calculate the frequency
// for this word existing in how many vectors as df. Using TF/DF as metric.
pub fn tf_idf<'a>(dialogue_datas: &Vec<Vec<&'a str>>) -> Vec<(&'a str, f64)> {
    let mut word_fre_dict: HashMap<&str, f64> = HashMap::new();
    let mut word_dia_fre_dict: HashMap<&str, f64> = HashMap::new();

    for a_dialogue in dialogue_datas {
        for word in a_dialogue {
            let fre = word_fre_dict.get(word).cloned().unwrap_or(0.0);
            word_fre_dict.insert(word, fre + 1.0);
        }
    }

    // calculate TF
    let all_num: f64 = word_fre_dict.values().into_iter().sum();
    for (k, v) in word_fre_dict.clone() {
        word_fre_dict.insert(k, v / all_num);
    }

    // calculate IDF
    for (k, v) in word_fre_dict.clone() {
        for adialogue in dialogue_datas {
            if adialogue.contains(&k) {
                let fre = word_dia_fre_dict.get(k).cloned().unwrap_or(0.0);
                word_dia_fre_dict.insert(k, fre + 1.0);
            }
        }
    }

    let all_dialogues = dialogue_datas.len() as f64;
    for (k, v) in word_dia_fre_dict.clone() {
        word_dia_fre_dict.insert(k, v / (all_dialogues + 1.0).ln());
    }

    // calculate the final TF*IDF results.
    let mut word_tfidf_dt = HashMap::<&str, f64>::new();
    for (k, v) in word_fre_dict {
        word_tfidf_dt.insert(k, v * word_dia_fre_dict.get(k).unwrap());
    }

    // sorted to a vec
    let mut tfidf_vec: Vec<(&str, f64)> = word_tfidf_dt.into_iter().collect();
    tfidf_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    return tfidf_vec;
}

// given tokenized dialogue corpus, return a rank of words depending on their
// importances under this corpus. Here TF_DF algorithm was used, which calculate
// the text frequency of a word in a vector as tf, and then calculate the frequency
// for this word existing in how many vectors as df. Using TF*DF as metric.
pub fn tf_df<'a>(dialogue_datas: &Vec<Vec<&'a str>>) -> Vec<(&'a str, f64)> {
    let mut word_fre_dict: HashMap<&str, f64> = HashMap::new();
    let mut word_dia_fre_dict: HashMap<&str, f64> = HashMap::new();

    for a_dialogue in dialogue_datas {
        for word in a_dialogue {
            let fre = word_fre_dict.get(word).cloned().unwrap_or(0.0);
            word_fre_dict.insert(word, fre + 1.0);
        }
    }

    // calculate TF
    let all_num: f64 = word_fre_dict.values().into_iter().sum();
    for (k, v) in word_fre_dict.clone() {
        word_fre_dict.insert(k, v / all_num);
    }

    // calculate IDF
    for (k, v) in word_fre_dict.clone() {
        for adialogue in dialogue_datas {
            if adialogue.contains(&k) {
                let fre = word_dia_fre_dict.get(k).cloned().unwrap_or(0.0);
                word_dia_fre_dict.insert(k, fre + 1.0);
            }
        }
    }

    let all_dialogues = dialogue_datas.len() as f64;
    for (k, v) in word_dia_fre_dict.clone() {
        word_dia_fre_dict.insert(k, (v / (all_dialogues + 1.0)).ln());
    }

    // calculate the final TF*DF results.
    let mut word_tfdf_dt = HashMap::<&str, f64>::new();
    for (k, v) in word_fre_dict {
        word_tfdf_dt.insert(k, v / word_dia_fre_dict.get(k).unwrap());
    }

    // sorted to a vec
    let mut tfdf_vec: Vec<(&str, f64)> = word_tfdf_dt.into_iter().collect();
    tfdf_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    return tfdf_vec;
}

// given a `dialogue` and the rank of words, return `topk` words in this dialogue.
pub fn find_keyword<'a, 'b>(
    dialogue: &Vec<&'a str>,
    sorted_words: &Vec<(&'b str, f64)>,
    topk: usize,
) -> Vec<&'a str> {
    let mut kw_vec: Vec<&str> = vec![];
    for w in dialogue {
        for (sw, fre) in sorted_words {
            if w == sw {
                kw_vec.push(w);
                if kw_vec.len() >= topk {
                    return kw_vec;
                }
            }
        }
        // break;
    }
    return kw_vec;
}

// take a dialogue example as input, return true if is chinese, else english.
fn is_chinese(dia: &UnlabeledDialogue) -> bool {
    true
    // false
}

//----------------------------------------
//     Basic Statistics Support
//----------------------------------------

// return the turn number distribution for a given unlabeled dialogue corpus.
pub fn get_turn_distribution(dialogues: &Vec<UnlabeledDialogue>) -> HashMap<i32, i64> {
    let mut turn_distri: HashMap<i32, i64> = HashMap::new();
    for dia in dialogues {
        let t = get_turn_num(&dia);
        turn_distri.insert(t, turn_distri.get(&t).unwrap_or(&0) + 1);
    }

    turn_distri
}

// Given an unlabeleed dialogue return its turn numebrs.
fn get_turn_num(dia: &UnlabeledDialogue) -> i32 {
    let turns = &dia.contents;
    let turn_num = turns.len().to_i32().unwrap();
    turn_num
}
