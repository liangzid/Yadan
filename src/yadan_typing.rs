// the module `core_toda` is the core components of task-oriented dialogue
// system analisis tool, which consists of the definition of Domain,Value,
// Belief States, Database retrievaled results, dialogue acts as well as
// the definition of dialogue and dialogues.
// core_toda has defined the standard data format of tod, and then gvies
// some useful methods to applied it.
// pub mod core_toda{

use std::collections::btree_set::Union;
use std::collections::HashMap;
use std::fmt::Result;
use std::fs::File;
use std::hash::Hash;
use std::io::BufRead;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use serde_json::{Error, Map, Value};

use crate::toda_database::DBSlot;
// use serde_derive::{Serialize,Deserialize};

// --------------------------------------------------------------------
// below is the basic data structure of task-oriented dialogue systems.
// --------------------------------------------------------------------

pub type History = Vec<String>;

//// Data Structure for Parsing TOD Schema.
pub type Domain = Option<String>;
pub type SDomain = String;
pub type Intent = String;

pub type SSlot = String;
pub type SValue = String;
pub type DialogueValue = String;

pub type Act = (SDomain, Intent, SSlot, SValue);
pub type SAct = String; // note that here the Structure of Act is not the its finnally shape.
                        // we define Schema as a hiearchical structure, which consists of
                        // the key information of Database.
pub type SqlQ = String;

pub type TurnState = Vec<Act>;
pub type Belief = HashMap<Domain, HashMap<String, DialogueValue>>;

pub type DatabaseMatching = HashMap<Domain, i32>;

//// parse database with Json.
pub type OriginDataBaseItem = HashMap<Option<String>, Value>;
pub type EncodedDataBaseItem = HashMap<String, String>;
pub type vec_str = Vec<String>;
pub type vec_f32 = Vec<f32>;
#[derive(Debug, Serialize, Deserialize)]
pub enum DatabaseValue {
    Null,
    String(String),
    vec_str(vec_str),
    vec_f32(vec_f32),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Dialogues {
    dialogues: Vec<Dialogue>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Dialogue {
    name: String,
    goal: Value,
    items: Vec<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
enum Item {
    ItemBySystem(ItemBySystem),
    ItemByUser(ItemByUser),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ItemBySystem {
    speaker: String,
    text: String,
    dialogue_act: Vec<Act>,
    active_domain: Domain,
    belief: Belief,
    booked_domains: Vec<Domain>,
    delexicalised_text: String,
    database: DatabaseMatching,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ItemByUser {
    speaker: String,
    text: String,
    dialogue_act: Vec<Act>,
}

// A DialogueTurn is a pair a dialogues, called user utterance and system responses.
pub type DialogueTurn = (String, String);

// Role is a enum type variable, that own User and System. The default value is User.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    User,
    System,
}

impl Default for Role {
    fn default() -> Role {
        let results: Role = Role::User;
        results
    }
}

// An unlabeledDialogue is a sequence of dialogue turns.
// It consists of two attributes: 1. contents, which is
// a vector of DialogueTurn, to store raw dialogues, and
// 2. first_role to descriminate the first sentence in
// this dialogue is User or System. Normally, if it is
// user, then all dialogueturn will start with a User utternace.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct UnlabeledDialogue {
    pub first_role: Role,
    pub contents: Vec<DialogueTurn>,
}

// transfer a structure history to string.
pub fn encode_history(history: &Vec<String>) -> String {
    let mut res = String::from("");
    for per_u in history {
        res += per_u;
    }
    res
}

// transfer a structure history to string.
pub fn decode_history(history: &str) -> Vec<String> {
    let mut res: Vec<String> = vec![];
    let split_his = history.split(";");
    for per_u in split_his {
        if per_u == "" {
            continue;
        } else {
            res.push(per_u.to_string());
        }
    }
    res
}

// transfer a structure acts to string.
pub fn encode_dialogue_act(acts: &Vec<Act>) -> String {
    let mut encoded_acts = String::from("").to_owned();
    let seperator = String::from(",");
    let big_sep = String::from(";");
    for act in acts {
        let (intent, domain, slot, value) = act;
        let encoded_act: String = intent.to_owned()
            + &seperator
            + &domain
            + &seperator
            + &slot
            + &seperator
            + &value
            + &seperator
            + &big_sep;
        encoded_acts += &encoded_act;
    }
    encoded_acts
}

// transfer a structure belief states to string.
pub fn encode_belief(belief: &Belief) -> String {
    let mut encoded_belief = String::from("").to_owned();
    let domain_key_seq: &str = ":{";
    let key_seq: &str = ":";
    let key_end_seq: &str = ",";
    let end_seq: &str = "},";
    for (domain, per_belief) in belief.into_iter() {
        match domain {
            None => {}
            Some(domain) => {
                let mut per_encoded = domain.to_owned() + domain_key_seq;
                for (k, v) in per_belief.into_iter() {
                    let perthing = k.clone() + key_seq + &v + key_end_seq;
                    per_encoded += &perthing;
                }
                per_encoded += &end_seq;
                encoded_belief += &per_encoded;
            }
        }
    }
    encoded_belief
}

// transfer a structure database into a string.
pub fn encode_database(db: &DatabaseMatching) -> String {
    let mut encoded_db: String = "".to_owned();
    for (domain, num) in db.into_iter() {
        match domain {
            None => {}
            Some(domain) => {
                let per = domain.clone() + &" " + &num.to_string() + &" matched,";
                encoded_db += &per;
            }
        }
    }
    encoded_db
}

// transfer a string acts into a acts structure.
pub fn decode_dialogue_act(e_acts: &str) -> Vec<Act> {
    let mut acts: Vec<Act> = vec![];
    let half_acts = e_acts.clone().split(";");
    // println!("{:?}",half_acts);
    for act in half_acts {
        if act == "" {
            continue;
        } else {
            let mut per_components = act.split(",");
            acts.push((
                per_components.nth(0).unwrap_or_default().to_string(),
                per_components.nth(0).unwrap_or_default().to_string(),
                per_components.nth(0).unwrap_or_default().to_string(),
                per_components.nth(0).unwrap_or_default().to_string(),
            ));
        }
    }
    acts
}

// transfer a string belief into a belief states structure.
pub fn decode_belief(e_belief: &str) -> Belief {
    let mut belief: HashMap<Domain, HashMap<String, DialogueValue>> = HashMap::new();
    let t_belief = e_belief.clone().split("},");
    for b in t_belief {
        let mut per_domain_belief: HashMap<String, DialogueValue> = HashMap::new();
        if b == "" {
            continue;
        } else {
            let domain: String = b.split(":{").nth(0).unwrap_or_default().to_string();
            let kvs = b.split(":{").nth(1).unwrap_or_default().to_string();
            let kvs = kvs.split(",");
            for perkv in kvs {
                if perkv == "" {
                    continue;
                } else {
                    let k = perkv.split(":").nth(0).unwrap_or_default().to_string();
                    let v = perkv.split(":").nth(1).unwrap_or_default().to_string();
                    per_domain_belief.insert(k.clone(), v.clone());
                }
                belief.insert(Some(domain.clone()), per_domain_belief.clone());
            }
        }
    }
    belief
}

pub fn Belief2DBSlots(b:&Belief)->HashMap<String,Vec<DBSlot>>{
    let mut db_slots:HashMap<String,Vec<DBSlot>>=HashMap::new();
    for (domain,per_state) in b.into_iter(){
	match domain{
	    Some(d)=>{
		let mut dbs:Vec<DBSlot>=vec![];
		let mut this_vec:Vec<String>=vec![];
		for (s,v) in per_state{
		    this_vec.push(s.clone());
		    dbs.push(DBSlot{name:s.clone(), slot_type:String::from("STR")});
		}
		db_slots.insert(d.clone(),dbs);
	    }
	    None=>{continue;}
	}
    }
    db_slots
}

// transfer a string acts into a acts structure.
pub fn decode_database(e_db: &str) -> DatabaseMatching {
    let mut db: HashMap<Domain, i32> = HashMap::new();
    let dbs = e_db.clone().split(",");
    for d in dbs {
        if d == "" {
            continue;
        } else {
            let info = d.split(" matched,").nth(0).unwrap_or_default().to_string();
            let domain: String = info.split(" ").nth(0).unwrap_or_default().to_string();
            let num: i32 = info
                .split(" ")
                .nth(1)
                .unwrap_or_default()
                .to_string()
                .parse::<i32>()
                .unwrap_or_default();
            db.insert(Some(domain), num);
        }
    }
    db
}

// ----- above data structure was declared done.
// ----- now we imply some of functions for it.
impl Dialogues {
    pub fn from_standard(filename: &str) -> Dialogues {
        // println!("{}",filename);
        let f = File::open(&filename).unwrap();
        // let datas:Value = serde_json::from_reader(&f).unwrap();
        // println!("{:?}",datas["dialogues"][0]);
        // let dialogues=datas.as_object().unwrap().clone();
        // let dialogues:Dialogues=dialogues;
        let dialogues: Dialogues = serde_json::from_reader(f).unwrap();
        dialogues
    }

    // return all dialogue compontents in turn level, for training
    // language models or other usages.
    pub fn get_neuralpipeline_inputs(
        &self,
    ) -> Vec<(Vec<String>, Belief, DatabaseMatching, Vec<Act>, String)> {
        let mut results: Vec<(Vec<String>, Belief, DatabaseMatching, Vec<Act>, String)> = vec![];
        // let mut history_perturn:Vec<String>;

        for dialogue in &self.dialogues {
            // let mut contexts:Vec<String>;
            let mut history: Vec<String> = vec![];
            // let mut utterance:String;
            // let mut belief:Belief;
            for item in &dialogue.items {
                if let Some(field) = &item.get("belief") {
                    // println!("{:?}---",&item);
                    let system_item: ItemBySystem =
                        serde_json::from_str(&item.to_string()).unwrap();
                    // println!("system item is: {:?}",system_item);
                    // break;
                    results.push((
                        history.clone(),
                        system_item.belief,
                        system_item.database,
                        system_item.dialogue_act.clone(),
                        system_item.delexicalised_text.clone(),
                    ));
                    history.push("System: ".to_string() + &system_item.text + ";");
                } else {
                    let user_item: ItemByUser = serde_json::from_str(&item.to_string()).unwrap();
                    history.push("User: ".to_string() + &user_item.text + ";");
                }
                // break;
            }
        }
        return results;
    }

    // give the string level inputs, wich can be seen as the encoding of
    //`get_XXX_neuralpipeline_inputs`.
    //----------------------------
    //model type: soloist, ubar, and vanilla.
    //split_list: with length 5, 7, 6 respectively.
    //----------example----------------------
    // let model_type=model_type.unwrap_or("vanilla");
    // let default_sl=vec!["<|boh|>".to_string(),"<|bob|>".to_string(),
    //                                 "<|eob|>".to_string(),"<|eod|>".to_string(),
    //                                 "<|eoa|>".to_string(),"<|eos|>".to_string()];
    pub fn get_neuralpipeline_string_intputs(
        &self,
        model_type: &str,
        split_list: &Vec<String>,
    ) -> Vec<String> {
        let mut results: Vec<String> = vec![];
        if model_type == "soloist" {
            let running_results = self.get_soloist_neuralpipeline_inputs();
            for perreuslt in running_results {
                let (history, bs, db, response) = perreuslt;
                let per_str: String = split_list[0].clone()
                    + &encode_history(&history)
                    + &split_list[1]
                    + &encode_belief(&bs)
                    + &split_list[2]
                    + &encode_database(&db)
                    + &split_list[3]
                    + &response
                    + &split_list[4];
                results.push(per_str);
            }
        } else if model_type == "ubar" {
            let running_results = self.get_UBAR_neuralpipeline_inputs();
            for perreuslt in running_results {
                let mut this_turn_results = String::from(split_list[0].clone());
                for (i, history) in perreuslt.iter().enumerate() {
                    let (utterance, bs, db, acts, response) = history;
                    if i == perreuslt.len() - 1 {
                        let sub_turn = split_list[2].clone()
                            + utterance
                            + &split_list[3]
                            + &encode_belief(bs)
                            + &split_list[4]
                            + &encode_database(db)
                            + &split_list[5]
                            + &encode_dialogue_act(acts)
                            + response
                            + &split_list[6];
                        this_turn_results += &sub_turn;
                    } else {
                        let sub_turn = (*utterance).clone()
                            + &encode_belief(bs)
                            + &encode_database(db)
                            + &encode_dialogue_act(acts)
                            + response
                            + &split_list[1];
                        this_turn_results += &sub_turn;
                    }
                }
                results.push(this_turn_results);
            }
        } else {
            let running_results = self.get_neuralpipeline_inputs();
            for perreuslt in running_results {
                let (history, bs, db, acts, response) = perreuslt;
                let per_str: String = split_list[0].clone()
                    + &encode_history(&history)
                    + &split_list[1]
                    + &encode_belief(&bs)
                    + &split_list[2]
                    + &encode_database(&db)
                    + &split_list[3]
                    + &encode_dialogue_act(&acts)
                    + &split_list[4]
                    + &response
                    + &split_list[5];
                results.push(per_str);
            }
        }
        results
    }

    // This function will return a soloist neuralpipeline inputs, which
    // have no dialogue actions, compared to vanilla neuralpipeline inputs.
    pub fn get_soloist_neuralpipeline_inputs(
        &self,
    ) -> Vec<(Vec<String>, Belief, DatabaseMatching, String)> {
        let mut results: Vec<(Vec<String>, Belief, DatabaseMatching, String)> = vec![];
        // let mut history_perturn:Vec<String>;

        for dialogue in &self.dialogues {
            // let mut contexts:Vec<String>;
            let mut history: Vec<String> = vec![];
            // let mut utterance:String;
            // let mut belief:Belief;
            for item in &dialogue.items {
                if let Some(field) = &item.get("belief") {
                    // println!("{:?}---",&item);
                    let system_item: ItemBySystem =
                        serde_json::from_str(&item.to_string()).unwrap();
                    // println!("system item is: {:?}",system_item);
                    // break;
                    results.push((
                        history.clone(),
                        system_item.belief,
                        system_item.database,
                        system_item.delexicalised_text.clone(),
                    ));
                } else {
                    let user_item: ItemByUser = serde_json::from_str(&item.to_string()).unwrap();
                    history.push(user_item.text);
                }
                // break;
            }
        }
        return results;
    }

    // This function will return a UBAR style neural pipeline inputs, where
    // each item own past turn history, and `utterance, belief, db,acts,responses`
    // in this turn.
    pub fn get_UBAR_neuralpipeline_inputs(
        &self,
    ) -> Vec<Vec<(String, Belief, DatabaseMatching, Vec<Act>, String)>> {
        let mut results: Vec<Vec<(String, Belief, DatabaseMatching, Vec<Act>, String)>> = vec![];

        for dialogue in &self.dialogues {
            let mut history: Vec<(String, Belief, DatabaseMatching, Vec<Act>, String)> = vec![];
            let mut utterance: String = String::from("");
            for item in &dialogue.items {
                if let Some(field) = &item.get("belief") {
                    // println!("{:?}---",&item);
                    let system_item: ItemBySystem =
                        serde_json::from_str(&item.to_string()).unwrap();
                    // println!("system item is: {:?}",system_item);
                    // break;
                    history.push((
                        utterance.clone(),
                        system_item.belief,
                        system_item.database,
                        system_item.dialogue_act,
                        system_item.delexicalised_text.clone(),
                    ));
                    results.push(history.clone());
                } else {
                    let user_item: ItemByUser = serde_json::from_str(&item.to_string()).unwrap();
                    utterance = user_item.text;
                }
                // break;
            }
        }
        return results;
    }
}

// }
