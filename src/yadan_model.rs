use crate::yadan_typing::{Intent,SDomain,SSlot,
			   DialogueValue,TurnState,SqlQ,
			   History,Belief,Act};
use std::error::Error;
use std::path::PathBuf;
use std::collections::HashMap;

use tch::{nn, Device, Tensor};

use rust_tokenizers::tokenizer::Gpt2Tokenizer;
use rust_bert::gpt2::{GPT2LMHeadModel, Gpt2Config,GPT2Generator};
use rust_bert::pipelines::generation_utils::{
    GenerateConfig, GenerateOptions, LanguageGenerator,
};
use rust_bert::resources::{LocalResource, Resource};
use rust_bert::Config;

use async_trait::async_trait;

pub trait utterance_nlu {
    fn utter2intents(&self,utter:&str)->Vec<Intent>{let x:Vec<Intent>=vec![]; x}
    fn utter2domains(&self,utter:&str)->Vec<SDomain>{let x:Vec<SDomain>=vec![];x}
    fn utter2svpairs(&self,utter:&str)->Vec<(SSlot,DialogueValue)>
    {
	let x:Vec<(SSlot,DialogueValue)>=vec![];
     x}
    fn utter2turn_state(&self,utter:&str)->TurnState{let x:TurnState=vec![];x}
    fn utter2sqlq(&self,utter:&str)->SqlQ{String::from("")}
}

pub trait history_nlu {
    fn his2intents(&self,history:&History)->Vec<Intent>{let x:Vec<Intent>=vec![];x}
    fn his2domains(&self,history:&History)->Vec<SDomain>{let x:Vec<SDomain>=vec![];x}
    fn his2svpairs(&self,history:&History)->Vec<(SSlot,DialogueValue)>
    {let x:Vec<(SSlot,DialogueValue)>=vec![];
     x}
    fn his2state(&self,history:&History,his_end_tk:Option<&str>,bs_end_tk:Option<&str>)->Belief{let mut x:Belief=HashMap::new();x}
    fn his2sqlq(&self,history:&History)->SqlQ{String::from("")}
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
    fn get_acts(&self,belief_state:&Belief)->Vec<Act>{let x:Vec<Act>=vec![];x}
}
pub trait Action_NLG{
    fn get_response(&self,acts:&Vec<Act>)->String{String::from("")}
}

pub trait Belief_NLG{
    fn get_response(&self,history:&History,belief:&Belief,
			  bb_stk:Option<&str>,be_stk:Option<&str>,
			  db_stk:Option<&str>,
			  re_stk:Option<&str>)->
	String{String::from("")}
}
pub trait Utter_NLG{
    fn get_response(&self,utterance:&str)->String{String::from("")}
}
pub trait History_NLG{
    fn get_response(&self,history:&History)->String{String::from("")}
}
pub trait Lexicalize {
    fn lexicalize(&self,delex_text:&str,belief:&Belief)->String{String::from("")}
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

