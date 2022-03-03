use std::collections::HashMap;
use std::fs::File;
use std::fs;
use std::io;
use std::path;
use std::path::PathBuf;
use zip;


pub fn type_of<T>(_: T) -> &'static str {
    std::any::type_name::<T>()
}

pub fn unzip(src_path:&str,dst_path:&str) -> String {

    let fname = std::path::Path::new(src_path);
    let file = fs::File::open(&fname).unwrap();

    let mut results="".to_owned();

    let mut archive = zip::ZipArchive::new(file).unwrap();

    for i in 0..archive.len() {
        let mut file = archive.by_index(i).unwrap();
	// println!("{:?}",&file);
        let outpath = match file.enclosed_name() {
            Some(path) => path.to_owned(),
            None => continue,
        };
	let str_opath=outpath.clone().into_os_string().into_string().unwrap();
	// println!("{:?}",&str_opath);
	// let outpath=std::path::Path::new(dst_path+outpath.unwrap().to_str());
	// break;
	let final_str:String=dst_path.to_owned()+&str_opath;
	let outpath=PathBuf::from(final_str.clone());

	if i==0{
	    results=final_str.clone();
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
