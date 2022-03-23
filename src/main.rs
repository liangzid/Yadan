use std::collections::btree_set::Union;
use std::fmt::Result;
use std::fs::File;
use std::collections::HashMap;
use std::hash::Hash;
use std::io::BufRead;
use std::str::FromStr;


// use ndarray_rand::rand::SeedableRng;
// extern crate rand;

#[macro_use]
extern crate serde;
extern crate serde_derive;
extern crate serde_json;

use serde::__private::de::StrDeserializer;
use serde_json::json;
use serde_json::{Value, Map, Error,};
use serde::{Serialize,Deserialize};

use async_std;

pub mod yadan_utils;
use yadan_utils::unzip;

// use serde_derive::{Serialize,Deserialize};

// pub mod core_toda;
// use core_toda::{Dialogues, Database};

pub mod yadan_typing;
use yadan_typing::{Dialogues, DatabaseMatching};

pub mod toda_database;
use toda_database::{fake_parse_database_json_file,fake_retrieval_entity_with_eq_attris};
use toda_database::{get_schema_from_sqlite,
		    retri_entit_for_attri,
		    DBSlot, retri_entit_for_attris};

pub mod toda_analysis;
use toda_analysis::{read_data_from_dir,
		    read_data_from_zip,
		    domain_mining,
                    get_turn_distribution,};

pub mod yadan_model;
use yadan_model::{SOLOIST};

fn test_yadan_typing(){
    // let filepath="/home/liangzi/multiwoz/multiwoz1.0/data.json";
    let filepath="/home/liangzi/multiwoz/soloist/multiwoz-2.1/train.json";
    let dialogues=Dialogues::from_standard(&filepath);
    // let npinputs=dialogues.get_soloist_neuralpipeline_inputs();
    // let npinputs=dialogues.get_UBAR_neuralpipeline_inputs();
    // let npinputs=dialogues.get_neuralpipeline_inputs();
    // let perinput=&npinputs[0];
    // let (history,belief,database,acts,response)=perinput;
    // println!("{}",encode_belief(&belief));
    // println!("{}",encode_database(&database));
    // println!("{}",encode_dialogue_act(&acts));
    // assert_eq!(belief, &decode_belief(&encode_belief(&belief)));
    // assert_eq!(database, &decode_database(&encode_database(&database)));
    // assert_eq!(acts, &decode_dialogue_act(&encode_dialogue_act(&acts)));
    let model_type="vanilla".to_string();
    let split_list=vec!["<|boh|>".to_string(),"<|bob|>".to_string(),
    "<|eob|>".to_string(),"<|eod|>".to_string(),
    "<|eoa|>".to_string(),"<|eos|>".to_string()];
    let test_str=dialogues.get_neuralpipeline_string_intputs(&model_type,&split_list);
    println!("{:?}",test_str);
}

async fn test_core_db(){
    // let fpath="/home/liangzi/multiwoz/multiwoz1.0/attraction_db.json";
    // let data=fake_parse_database_json_file(&fpath);
    // println!("{:?}",data[0]);
    // let attributs=HashMap::from([("area".to_string(),"centre".to_string()),
    // ("pricerange".to_string(),"free".to_string())]);
    // let results=fake_retrieval_entity_with_eq_attris(&data, &attributs);
    // println!("{:?}",results);
    // make_connect().await;

    let schema=get_schema_from_sqlite("/home/liangzi/test.db").await;
    // println!("{:?}",schema);

    let target_slot=DBSlot::find_Slot_with_name(&schema["COMPANY"].0,"NAME");
    let matched_results=retri_entit_for_attri("/home/liangzi/test.db",
					      "SELECT * FROM COMPANY WHERE AGE>25",
					      &target_slot).await;
    
    println!("{:?}",matched_results);


    let schema=get_schema_from_sqlite("/home/liangzi/test.db").await;
    // println!("{:?}",schema);

    let target_slot1=DBSlot::find_Slot_with_name(&schema["COMPANY"].0,"NAME");
    let target_slot2=DBSlot::find_Slot_with_name(&schema["COMPANY"].0,"ID");
    let matched_results=retri_entit_for_attris("/home/liangzi/test.db",
					      "SELECT * FROM COMPANY WHERE AGE>25",
					      &vec![target_slot1,target_slot2]).await;
    
    println!("{:?}",matched_results);

}

fn test_toda_analysis(){
    // let dpath:&str="/mnt/d/data/unlabeled_dialogue_for_toda";
    // let dialogues=read_data_from_dir(dpath);
    // println!("{:?}",dialogues[0]);

    let fpath:&str="/mnt/d/data/unlabeled.zip";
    let dialogues=read_data_from_zip(fpath);
    // println!("{:?}",dialogues[0]);

    // let clusters=domain_mining(&dialogues,"/mnt/d/code/wiki-word2vec-chinese.bin");

    // let distirbutions= get_turn_distribution(&dialogues);
    // println!("distributions: {:?}",distirbutions);

}

fn test_yadan_model(){
    // let pretrained_path="/home/zliang/code/distilgpt2/";
    let pretrained_path="/home/zliang/backinference/augpt-bigdata/";
    let soloist_model:SOLOIST=SOLOIST::init(pretrained_path);

    let prefix_his:&str="I would like a taxi from Saint John's college to Pizza Hut Fen Ditton.";
    let result:String=soloist_model.forward(prefix_his);
    println!("{}",result);

}

#[async_std::main]
async fn main() {
    // test_yadan_typing();
    // test_core_db().await;
    // test_toda_analysis();
    test_yadan_model();
}
