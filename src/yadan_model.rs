use yadan::yandan_typing::{Intent,SDomain,SSlot,
			   DialogueValue,TurnState,SqlQ,
			   History,Belief,Act};
use std::path::PathBuf;

use torch_sys::{nn,Device,Tensor};

use rust_tokenizers::tokenizer::Gpt2Tokenizer;
use rust_bert::gpt2::{GPT2LMHeadModel, Gpt2Config,GPT2Generator};
use rust_bert::pipelines::generation_utils::{
    GenerateConfig, GenerateOptions, LanguageGenerator,
};
use rust_bert::resources::{LocalResource, Resource};
use rust_bert::Config;

pub trait utterance_nlu {
    fn utter2intents(&self,utter:&str)->Vec<Intent>{}
    fn utter2domains(&self,utter:&str)->Vec<SDomain>{}
    fn utter2svpairs(&self,utter:&str)->Vec<(SSlot,DialogueValue)>{}
    fn utter2turn_state(&self,utter:&str)->TurnState{}
    fn utter2sqlq(&self,utter:&str)->SqlQ{}
}

pub trait history_nlu {
    fn his2intents(&self,history:&History)->Vec<Intent>{}
    fn his2domains(&self,history:&History)->Vec<SDomain>{}
    fn his2svpairs(&self,history:&History)->Vec<(SSlot,DialogueValue)>{}
    fn his2state(&self,history:&History)->Belief{}
    fn his2sqlq(&self,history:&History)->SqlQ{}
}

pub type UtterNLU=dyn utterance_nlu;
pub type HisNLU=dyn history_nlu;

enum NLUModel{
    UtterNLU(Box<UtterNLU>),
    HisNLU(Box<HisNLU>),
}

pub trait dst{
    fn update(&self,belief:&mut Belief,new_state:&TurnState){}

}

pub trait Decision {
    fn get_acts(&self,belief_state:&Belief)->Vec<Act>{}
}
pub trait Action_NLG{
    fn get_response(&self,acts:&Vec<Act>)->String{String::from("")}
}
pub trait Belief_NLG{
    fn get_response(&self,belief:&Belief)->String{String::from("")}
}
pub trait Utter_NLG{
    fn get_response(&self,utterance:&str)->String{String::from("")}
}
pub trait History_NLG{
    fn get_response(&self,history:&History)->String{String::from("")}
}
pub trait Lexicalize {
    fn lexicalize(&self,delex_text:&str)->String{String::from("")}
}

pub type dstt=dyn dst;
pub type Decisiont=dyn Decision;
pub type ANLG=dyn Action_NLG;
pub type BNLG=dyn Belief_NLG;
pub type UNLG=dyn Utter_NLG;
pub type HNLG=dyn History_NLG;
pub type Lexi=dyn Lexicalize;

pub struct Pipeline4<'a>{
    description:&'a str,
    lexicalizer: Box<dyn Lexicalize>,

    nlu_module: Box<UtterNLU>,
    dst_module: Box<dyn dst>,
    policy_module: Box<dyn Decision>,
    nlg_module: ANLG, 
}

pub struct Pipeline3<'a>{
    description:&'a str,
    lexicalizer: Box<dyn Lexicalize>,

    nlu_module: Box<HisNLU>,
    policy_module: Box<dyn Decision>,
    nlg_module: Box<dyn Action_NLG>, 
}

pub struct Pipeline2<'a>{
    description:&'a str,
    lexicalizer: Box<dyn Lexicalize>,

    nlu_module: Box<HisNLU>,
    nlg_module: Box<dyn Belief_NLG>, 
}

pub struct Pipeline1<'a>{
    description:&'a str,
    lexicalizer: Box<dyn Lexicalize>,
    nlg_module:Box<dyn History_NLG>, 
}


pub enum Pipeline<'a>{
    Pipeline4(Box<Pipeline4<'a>>),
    Pipeline3(Box<Pipeline3<'a>>),
    Pipeline2(Box<Pipeline2<'a>>),
    Pipeline1(Box<Pipeline1<'a>>),
}


pub struct SOLOIST{
    backbone: GPT2LMHeadModel,
    tokenizer: Gpt2Tokenizer,
    // lexicalizer: Box<dyn Lexicalize>,
}

// impl HisNLU for SOLOIST{

//     fn his2intents(&self,history:&History)->Vec<Intent>{
	
//     }

//     fn his2domains(&self,history:&History)->Vec<SDomain>{

	
//     }

//     fn his2svpairs(&self,history:&History)->Vec<(SSlot,DialogueValue)>{

//     }

//     fn his2state(&self,history:&History)->Belief{

//     }

//     fn his2sqlq(&self,history:&History)->SqlQ{

//     }

// }

impl SOLOIST{

    pub fn init(pretrained_path:&str)->SOLOIST{

	// loading tokenizer  and pretrained models.
	// Noting: here the model can be checkpoints or universal pretrained models.
	let config_path=pretrained_path.to_owned()+"/config.json";
	let vocab_path=pretrained_path.to_owned()+"/vocab.txt";
	let model_path=pretrained_path.to_owned()+"/model.ot";

	let config_resource=Resource::Local(LocalResource{
	    local_path: PathBuf::from(&config_path)});
	let vocab_resource=Resource::Local(LocalResource{
	    local_path: PathBuf::from(&config_path)});
	let merges_resource=Resource::Local(LocalResource{
	    local_path: PathBuf::from(&config_path)});
	let weights_resource=Resource::Local(LocalResource{
	    local_path: PathBuf::from(&config_path)});

	let config_path=config_resource.get_local_path();
	let vocab_path=vocab_resource.get_local_path();
	let merges_path=merges_resource.get_local_path();
	let weights_path=weights_resource.get_local_path();

	let device= Device::cuda_if_available();
	let mut vs = nn::VarStore::new(device);

	let tokenizer = Gpt2Tokenizer::from_file(
	    vocab_path.to_str().unwrap(),
	    merges_path.to_str().unwrap(),
	    true,
	);
	let config=Gpt2Config::from_file(config_path);
	let backbone = GPT2LMHeadModel::new(&vs.root(),&config);
	vs.load(weights_path);

	// loading lexicalizer
	let soloist= SOLOIST { backbone:backbone, tokenizer:tokenizer };
	soloist
    }

    pub fn forward(self,prefix_his:&str)->String{
	let generate_options=GenerateOptions{
            min_length: Some(32),
            max_length:Some(128),
            output_scores:false,
	    ..Default::default()
            // prefix_allowed_tokens_fn: Some(&force_one_paragraph);
	};

	let output = self.backbone.generate(
	    Some(&[prefix_his]),
	    Some(generate_options),
	);

	let sequence:String=output.at(0).unwrap().text;
	return sequence;
    }
}

// impl Belief_NLG for SOLOIST{
//     fn get_response(&self, belief:&Belief)-> String{
	

//     }
// }

// impl Lexicalize for SOLOIST{
//     fn lexicalize(&self,delex_text:&str)->String{

//     }

// }


