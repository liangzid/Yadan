use yadan::yandan_typing::*;

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

pub type UtterNLU= utterance_nlu;
pub type HisNLU= history_nlu;

enum NLUModel{
    UtterNLU(UtterNLU),
    HisNLU(HisNLU),
}

pub trait dst{
    fn update(&self,belief:&mut Belief,new_state:&TurnState){}

}

pub trait Decision {
    fn get_acts(&self,belief_state:&Belief)->Vec<Act>{}
}

pub trait Action_NLG{
    fn get_response(&self,acts:&Vec<Act>)->String{}
}

pub trait Belief_NLG{
    fn get_response(&self,belief:&Belief)->String{}
}

pub trait Utter_NLG{
    fn get_response(&self,utterance:&str)->String{}
}

pub trait History_NLG{
    fn get_response(&self,history:&History)->String{}
}

pub trait Lexicalize {
    fn lexicalize(&self,delex_text:&str)->String{}
    
}

pub struct Pipeline4{
    description:&str,
    lexicalizer: impl lexicalize,

    nlu_module: UtterNLU,
    dst_module: impl dst,
    policy_module: impl Decision,
    nlg_module: impl Action_NLG, 
}

pub struct Pipeline3{
    description:&str,
    lexicalizer: impl lexicalize,

    nlu_module: HisNLU,
    policy_module: impl Decision,
    nlg_module: impl Action_NLG, 
}

pub struct Pipeline2{
    description:&str,
    lexicalizer: impl lexicalize,

    nlu_module: HisNLU,
    nlg_module: impl Belief_NLG, 
}

pub struct Pipeline1{
    description:&str,
    lexicalizer: impl lexicalize,

    nlg_module: impl History_NLG, 
}


enum Pipeline{
    Pipeline4(Pipeline4),
    Pipeline3(Pipeline3),
    Pipeline2(Pipeline2),
    Pipeline1(Pipeline1),
}

