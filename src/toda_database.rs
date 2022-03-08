// use std::collections::btree_set::Union;
use std::fmt::Result;
use std::fs::File;
use std::collections::HashMap;
use std::io::BufRead;

use serde::__private::de::StrDeserializer;
use serde_json::json;
use serde_json::{Value, Map, Error,};
use serde::{Serialize,Deserialize};
// use serde_derive::{Serialize,Deserialize};

// use sea_orm::{Statement};
// use sqlx::Database;
// use sqlx::sqlite::SqlitePoolOptions;
use sqlx::{Connection,Executor,query,Row};
use futures::TryStreamExt;

// pub mod yadan_typing;
use yadan::yadan_typing::{SDomain,SAct,OriginDataBaseItem,EncodedDataBaseItem,
			  vec_str,vec_f32,DatabaseValue};

// Slot own its name, e.g. `price`, `address`, and `salary`,
// and `slot_type` indicates its datatype in database.
#[derive(Debug,Default,Clone,)]
pub struct DBSlot {
    name:String,
    slot_type:String,
}

pub type Schema=HashMap<SDomain,(Vec<DBSlot>,Vec<SAct>)>;

impl DBSlot{
    // give a Sequence of Slots and the search name, return
    // the DBSlot I want to finded. If there are not one Slots
    // matched, it will return the first. And it will return
    // a default results if there is no matched DBSlot.
    pub fn find_Slot_with_name(slots:&Vec<DBSlot>,name:&str)->DBSlot{
	let mut target=DBSlot::default();
	for s in slots{
	    if s.name==name{
		target=s.clone();
	    }
	}
	target
    }

    // Given a SQL query that try to get the describle information
    // of a table, this function will return all found slots from
    // SQL query results.
    // For example:`SELECT sql FROM sqlite_master
    //              WHERE type = 'table' AND tbl_name = 'table_name'`
    // will return a column named sql, which is the create table
    // sql language of table_name. this function parses such SQL sentence.
    pub fn parse_SQLite(sql_results:&str)->Vec<DBSlot>{
	let mut res:Vec<DBSlot>=vec![];
	let slices=sql_results.split("\n");
	for slice in slices.into_iter(){
	    
	    if slice.contains("CREATE TABLE") || slice.contains("ceate table"){
		continue;
	    }
	    if slice.contains(")") && !slice.contains("("){
		continue;
	    }
	    if slice==""{
		continue;
	    }

	    let things=slice.split(" ");
	    let mut elements:Vec<&str>=vec![];
	    for thing in things{
		if thing!=""{
		    elements.push(&thing);
		}
	    }
	    res.push(DBSlot{name: elements.get(0).unwrap().to_string(),
			  slot_type:elements.get(1).unwrap().to_string()});
	}
	res
    }
}

// given a `url` of target link datbase, this function
// will return the schema of all tables from this url.
// above procedure is async, which means you should
// add `.await` when you use this function. 
pub async fn get_schema_from_sqlite(url:&str)->Schema{

    let mut finnally_schema:Schema=HashMap::new();
    let mut domains:Vec<SDomain>=vec![];
    let empty_acts:Vec<SAct>=vec![];

    // make connection
    let mut conn1=sqlx::SqliteConnection::connect(url).await.unwrap();
    let mut conn2=sqlx::SqliteConnection::connect(url).await.unwrap();

    //find domains, which is the table name.
    let mut table_query_res=sqlx::query("SELECT tbl_name FROM sqlite_master WHERE type='table'")
	.fetch(&mut conn1);
    while let Some(table_name_row)=table_query_res.try_next().await.unwrap(){

	let domain_name:&str=table_name_row.try_get("tbl_name").unwrap();
	// println!("{}",domain_name);
	domains.push(domain_name.to_string());
    }

    //find the schema of each domain, which can contrut the task-oriented dialogue schema
    for domain in domains{
	// if domain==""{continue;}
	
	let mut slots:Vec<DBSlot>=vec![];
	let q="SELECT sql FROM sqlite_master WHERE type = 'table' AND tbl_name =".to_owned()+&"'"+&domain+&"'";
	let mut schemas=sqlx::query(&q).fetch(&mut conn2);
	    
	while let Some(per_attribute)=schemas.try_next().await.unwrap(){
	    // println!("{:?}",per_attribute);

	    let slot_name:&str=per_attribute.try_get("sql").unwrap();
	    // println!("{:?}",slot_name);
	    slots.append(&mut DBSlot::parse_SQLite(slot_name));
	}
	
	finnally_schema.insert(domain.to_string(),(slots,empty_acts.clone()));
    }
    

    // println!("{:?}",table_query_res);
    
    // let schema_information=conn.execute("SELECT * FROM COMPANY1.TABLES").await.unwrap();
    // println!("{:?}",&schema_information);

    finnally_schema
}

// given the target database url `db_url`, and the expected SQL query
// sentence `sql_sent`, as well as the attribute `attri` you want to
// return, this function will return a sequence of values about `attri`
// that matched the query.
// above procedure is async, which means you should
// add `.await` when you use this function. 
pub async fn retri_entit_for_attri(db_url:&str,sql_sent:&str,attri:&DBSlot)
				      ->Vec<String>{
    let mut matched_results:Vec<String>=vec![];
    
    // make connection
    let mut conn1=sqlx::SqliteConnection::connect(db_url).await.unwrap();
    // let mut conn2=sqlx::SqliteConnection::connect("/home/liangzi/test.db").await.unwrap();

    // matched related rows
    let mut matched_rows=sqlx::query(sql_sent)
	.fetch(&mut conn1);
    while let Some(row)=matched_rows.try_next().await.unwrap(){
	let mut attribute_results:String="".to_owned();
	if &*attri.slot_type == "INT"{
	    let attribute:i32=row.try_get(&(*attri.name)).unwrap();
	    attribute_results=attribute.to_string();
	}
	else if &*attri.slot_type=="REAL" {
	    let attribute:f32=row.try_get(&(*attri.name)).unwrap();
	    attribute_results=attribute.to_string();
	}
	else if &*attri.slot_type=="BOOL" {
	    let attribute:bool=row.try_get(&(*attri.name)).unwrap();
	    attribute_results=attribute.to_string();
	}
	else{
	    let attribute:&str=row.try_get(&(*attri.name)).unwrap();
	    attribute_results=attribute.to_string();

	}
	matched_results.push(attribute_results.to_string());
    }
    matched_results
}

// given the target database url `db_url`, and the expected SQL query
// sentence `sql_sent`, as well as the attributes list `attris` you want
// to return, this function will return a recodes that satisfy your
// query. the shape of returned recodes is a sequences of a hashmap, will
// the hashmap acts as a dictionary for filtered row info., and all hashmaps
// in Vec have the same key, which is attris.name
pub async fn retri_entit_for_attris(db_url:&str,sql_sent:&str,attris:&Vec<DBSlot>)
				      ->Vec<HashMap<String,String>>{
    let mut matched_results:Vec<HashMap<String,String>>=vec![];
    
    // make connection
    let mut conn1=sqlx::SqliteConnection::connect(db_url).await.unwrap();
    // let mut conn2=sqlx::SqliteConnection::connect("/home/liangzi/test.db").await.unwrap();

    // matched related rows
    let mut matched_rows=sqlx::query(sql_sent)
	.fetch(&mut conn1);
    while let Some(row)=matched_rows.try_next().await.unwrap(){
	let mut match_per_row:HashMap<String,String>=HashMap::new();
	for attribute in attris{
	    let mut attribute_results:String="".to_owned();
	    if &*attribute.slot_type == "INT"{
		let attribute:i32=row.try_get(&(*attribute.name)).unwrap();
		attribute_results=attribute.to_string();
	    }
	    else if &*attribute.slot_type=="REAL" {
		let attribute:f32=row.try_get(&(*attribute.name)).unwrap();
		attribute_results=attribute.to_string();
	    }
	    else if &*attribute.slot_type=="BOOL" {
		let attribute:bool=row.try_get(&(*attribute.name)).unwrap();
		attribute_results=attribute.to_string();
	    }
	    else{
		let attribute:&str=row.try_get(&(*attribute.name)).unwrap();
		attribute_results=attribute.to_string();

	    }
	    match_per_row.insert(attribute.name.clone(), attribute_results.to_string());
	}
	// println!("{}",attribute_results);
	matched_results.push(match_per_row);
    }
    matched_results
}

// ----------------------------------------------------------------------
// BLEW Functions were specifically designed for MultiWOz Json Style DBs.
// ----------------------------------------------------------------------


// Parse fake databases, and return a Dictionary Sequence.
// Note that EncodedDataBaseItem is the HashMap, and Vec<x> is the structure
// of those db files. 
pub fn fake_parse_database_json_file(fpath:&str)->Vec<EncodedDataBaseItem>{
    let mut results:Vec<EncodedDataBaseItem>=vec![];
    let f=File::open(&fpath).unwrap();
    // let datas:Value = serde_json::from_reader(&f).unwrap();
    // println!("{:?}",datas["dialogues"][0]);
    // let dialogues=datas.as_object().unwrap().clone();
    // let dialogues:Dialogues=dialogues;
    let results_read:Vec<OriginDataBaseItem>=serde_json::from_reader(f).unwrap();
    for result in results_read{
        let mut dict:HashMap<String,String>=HashMap::new();
        for (attribute,v) in result{
            // let mut key="";
            let mut value=String::from("");
            if v.is_string(){
                value=v.as_str().unwrap().to_string();
            } 
            else{
                value=v.to_string(); 
            }
            // let mut value=v.to_string();
            match attribute{
                None=>{},
                Some(attribute)=>{
                    dict.insert(attribute, value);
                    }
            }
            }
        results.push(dict);
    }
    results
}

// retrieval the fake datrabases with slot-value.
// noted that this function only support for check which entities have the
// totally same slot-value pairs from parameter `attributes`, it cannot
// support some numeric operation like `> `. You should try to use SQL
// language from database related functions for more details.
pub fn fake_retrieval_entity_with_eq_attris(database:&Vec<EncodedDataBaseItem>,
        attributs:&HashMap<String,String>)->Vec<EncodedDataBaseItem>{
            let mut returned_entities:Vec<EncodedDataBaseItem>=vec![];
            
            for dataitem in database{
                let mut is_this_item_ok:u8=1;
                for (ak,av) in attributs{
                    let mut find_all_key_not_find_flag:u8=1;
                    let mut find_value_same_flag:u8=0;
                    for (dk,dv) in dataitem{
                        if dk==ak{
                            find_all_key_not_find_flag=0;
                            if dv==av{
                                find_value_same_flag=1;
                                break;
                            }
                            else{
                                break;
                            }

                        }
                    }
                    if find_all_key_not_find_flag==1{
                        is_this_item_ok=0;
                        break;
                    }
                    else{
                        if find_value_same_flag==0{
                            is_this_item_ok=0;
                            continue;
                        }

                    }
                }
                if is_this_item_ok==1{
                    returned_entities.push(dataitem.clone());
                }
            }
            returned_entities
}
