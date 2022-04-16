use crate::yadan_typing::{Intent,SDomain,SSlot,
			   DialogueValue,TurnState,SqlQ,
			   History,Belief,Act, DatabaseMatching, encode_database};

use crate::toda_database::{Schema, DBSlot, retri_entit_for_attri};

use crate::yadan_typing::{encode_history,
			  encode_belief,
			  decode_belief,
			  Belief2DBSlots,
};

use crate::yadan_error::YadanInferenceError;

use crate::yadan_model::{HisNLU, Belief_NLG, Lexicalize, history_nlu};

use std::error::Error;
use std::path::PathBuf;
use std::collections::HashMap;
use std::io;
use std::io::prelude::*;
extern crate simple_input;
use simple_input::input;
use num_traits::ToPrimitive;
use tch::{nn, Device, Tensor};

use rust_tokenizers::tokenizer::Gpt2Tokenizer;
use rust_bert::gpt2::{GPT2LMHeadModel, Gpt2Config,GPT2Generator};
use rust_bert::pipelines::generation_utils::{
    GenerateConfig, GenerateOptions, LanguageGenerator,
};
use rust_bert::resources::{LocalResource, Resource};
use rust_bert::Config;

use async_trait::async_trait;
use futures::{self,executor};

pub struct SOLOIST{
    generator: GPT2Generator,
    // schema: Schema,
    database_url:  String,
    max_length: i64,
    // backbone: GPT2LMHeadModel,
    // tokenizer: Gpt2Tokenizer,
    // lexicalizer: Box<dyn Lexicalize>,
}

impl history_nlu for SOLOIST{

    fn his2intents(&self,history:&History)->Vec<Intent>{let x:Vec<Intent>=vec![];x}

    fn his2domains(&self,history:&History)->Vec<SDomain>{let x:Vec<SDomain>=vec![];x}

    fn his2svpairs(&self,history:&History)->Vec<(SSlot,DialogueValue)>
    {let x:Vec<(SSlot,DialogueValue)>=vec![];
     x}

    fn his2state(&self,history:&History,his_end_tk:Option<&str>,bs_end_tk:Option<&str>)
		     ->Belief{

	// preprocess history for adding special tokens. 
	let shis:String=encode_history(history);
	self.context2state(&shis,his_end_tk,bs_end_tk)
    }
    

    fn his2sqlq(&self,history:&History)->SqlQ{String::from("")}

}

impl SOLOIST{

    pub fn init(pretrained_path:&str)->SOLOIST{

	// loading tokenizer  and pretrained models.
	// Noting: here the model can be checkpoints or universal pretrained models.
	let config_path=pretrained_path.to_owned()+"/config.json";
	let vocab_path=pretrained_path.to_owned()+"/vocab.json";
	let weights_path=pretrained_path.to_owned()+"/rust_model.ot";
	let merges_path=pretrained_path.to_owned()+"/merges.txt";

	let config_resource=Resource::Local(LocalResource{
	    local_path: PathBuf::from(&config_path)});
	let vocab_resource=Resource::Local(LocalResource{
	    local_path: PathBuf::from(&vocab_path)});
	let merges_resource=Resource::Local(LocalResource{
	    local_path: PathBuf::from(&merges_path)});
	let weights_resource=Resource::Local(LocalResource{
	    local_path: PathBuf::from(&weights_path)});

	// let config_path=config_resource.get_local_path().unwrap();
	// let vocab_path=vocab_resource.get_local_path().unwrap();
	// let merges_path=merges_resource.get_local_path().unwrap();
	// let weights_path=weights_resource.get_local_path().unwrap();

	// let device= Device::cuda_if_available();
	// let mut vs = nn::VarStore::new(device);

	// let tokenizer = Gpt2Tokenizer::from_file(
	//     vocab_path.to_str().unwrap(),
	//     merges_path.to_str().unwrap(),
	//     true,
	// ).unwrap();
	// let config=Gpt2Config::from_file(config_path);
	// let backbone = GPT2LMHeadModel::new(&vs.root(),&config);
	// vs.load(weights_path);

	// loading lexicalizer
	let gen_config=GenerateConfig{
	    model_resource:weights_resource,
	    config_resource:config_resource,
	    merges_resource:merges_resource,
	    vocab_resource:vocab_resource,
	    max_length: 500,
	    do_sample: true,
	    ..Default::default()
	};
	let mut gpt2_gen=GPT2Generator::new(gen_config).unwrap();
	let soloist= SOLOIST { generator:gpt2_gen,
			       database_url: "~/datasets/multiwoz-git/db/train_db.db".to_owned(),
			       max_length: 128 };
	return soloist
    }

    pub fn generate(&self,prefix_his:&str)->String{
	let generate_options=GenerateOptions{
            min_length: Some(32),
            max_length:Some(128),
            output_scores:false,
	    ..Default::default()
            // prefix_allowed_tokens_fn: Some(&force_one_paragraph);
	};

	let output = self.generator.generate(
	    Some(&[prefix_his]),
	    Some(generate_options),
	);

	let sequence:String=output[0].text.clone();
	return sequence;
    }

    pub fn inference_one_turn(&self,dialo_his:&Vec<String>)->(Belief,DatabaseMatching,String,String){
	// 1. dialogue state tracking, generating belief states from history.
	let current_belief:Belief=self.his2state(dialo_his, None, None);
	// 2. database retrieval.
	let db_matchs:DatabaseMatching=self.query_database(&current_belief);
	// 3. response generation based on above components.
	let delex_resp:String=self.get_response(dialo_his, &current_belief, None,None,None,None);
	let resp:String=self.lexicalize(&delex_resp, &current_belief);
	return (current_belief,db_matchs,delex_resp,resp);
    }

    pub fn interact(&self){
	// 0. init parameters
	let mut history:Vec<String>=vec![];
	while(true){
	    // 1. getting user input
	    let u_utter:String="User: ".to_owned()+&input(">>>User: ");
	    history.push(u_utter.clone());
	    // 2. make inference
	    let  (belief,db,dr,r)=self.inference_one_turn(&history);
	    // 3. rendering...
	    println!(">>>System: {}", &r);
	    history.push(r.clone());
	}
    }

    fn rule_based_sql(&self,domain:String, slots:&Vec<DBSlot>)->String{
	    let mut sql_sent:String=r#"SELECT sql FROM sqlite_master
 WHERE type = 'table' AND tbl_name='"#.to_owned()+&domain+&"'";
	    for dbs in slots{
		sql_sent= sql_sent+r#" AND "#+&dbs.name+&"'";
	    }
	sql_sent
    }

  fn query_database(&self, belief:&Belief)->DatabaseMatching{
	let name_slot=DBSlot{name:String::from("name"),slot_type:String::from("STR")};
	let mut db_matches:DatabaseMatching=HashMap::new();
	
	let dbslots:HashMap<String,Vec<DBSlot>>=Belief2DBSlots(belief);
	for (domain, slots) in dbslots{
	    let mut sql_sent:String=r#"SELECT sql FROM sqlite_master
 WHERE type = 'table' AND tbl_name='"#.to_owned()+&domain+&"'";
	    for dbs in slots{
		sql_sent= sql_sent+r#" AND "#+&dbs.name+&"'";
	    }
	    
	    let f=retri_entit_for_attri(&self.database_url,
							      &sql_sent,
							    &name_slot);
	    let queryresult:Vec<String>=executor::block_on(f);
	    db_matches.insert(Some(domain.clone()),queryresult.len().to_i32().unwrap());
	}
	db_matches
    }
  fn context2state(&self,shis:&str,his_end_tk:Option<&str>,bs_end_tk:Option<&str>)
		     ->Belief{

	// defining options
	// here I cannot any setting for eos token.
	let generate_options=GenerateOptions{
            min_length: Some(32),
            max_length:Some(self.max_length),
            output_scores:false,
	    ..Default::default()
	};

	// preprocess history for adding special tokens. 
	let final_his:String= match his_end_tk{
	    Some(stk)=> shis.to_owned() + stk,
	    None => shis.to_owned()+"<|bob|>=>Belief state : "
	};

	// make generation
	let output = self.generator.generate(
	    Some(&[final_his]),
	    Some(generate_options),
	);

	// post-process the generated sequence for extracting
	// belief states from outputs.
	let mut sequence:String=output[0].text.clone();
	let b_end=match bs_end_tk{
	    Some(x)=> x,
	    None=>"<|eob|>"
	};
	if sequence.contains(b_end){
	    sequence=sequence.split(b_end).nth(0).unwrap_or_default().to_string();
	}
	else{
	    sequence="".to_owned();
	}
	// if sequence==""{
	    // Err(YadanInferenceError{kind:String::from("Belief state generation"),message:String::from("The generated sequence is empty.")})
	// }
	// else{
	    let b:Belief=decode_belief(&sequence);
	    b
	// }
    }
}

impl Belief_NLG for SOLOIST{
fn get_response(&self,history:&History ,
		    belief:&Belief,
		    bb_stk:Option<&str>,be_stk:Option<&str>,
		    db_stk:Option<&str>,re_stk:Option<&str>)-> String {

	// defining options
	// here I cannot any setting for eos token.
	let generate_options=GenerateOptions{
            min_length: Some(32),
            max_length:Some(self.max_length),
            output_scores:false,
	    ..Default::default()
	};

	// preprocess history for adding special tokens. 
	let mut shis:String=encode_history(history);
	shis+=match bb_stk{
	    Some(stk)=> &stk,
	    None => &"<|bob|>=>Belief state : "
	};

	// add belief state
	shis=shis+&encode_belief(belief);
	shis+=match be_stk{
	    Some(stk)=> &stk,
	    None => &"<|eob|>"
	};

	let db_matches:DatabaseMatching=self.query_database(belief);
	let db_str:String=encode_database(&db_matches);
	shis=shis+&db_str;
	shis+=match db_stk{
	    Some(stk)=> &stk,
	    None => &"<|eod|>"
	};

	// make generation
	let output = self.generator.generate(
	    Some(&[&shis]),
	    Some(generate_options),
	);

	// post-process the generated sequence for extracting
	// belief states from outputs.
	let mut sequence:String=output[0].text.clone();
	let mut b_end=match re_stk{
	    Some(x)=> x,
	    None=>"<|endoftext|>"
	};
	if sequence.contains(b_end){
	    sequence=sequence.split(b_end).nth(0).unwrap_or_default().to_string();
	}
	else{
	    sequence="".to_owned();
	}
	// if sequence==""{
	//   Err(YadanInferenceError{kind:String::from("Response generation"),message:String::from("The generated sequence is empty.")})
	// }
	// else{
	    sequence
	// }
    }
}

impl Lexicalize for SOLOIST{
    fn lexicalize(&self,delex_text:&str,belief:&Belief)->String{
	let mut response:String=String::from(delex_text).clone();
	response
    }
}

