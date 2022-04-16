## YADAN
Yadan is a task-oriented dialogue engine aiming to help you constructing TOD quickly. It has following special characteristics:
* Data/Action/Knowledge Centered. For yadan, you must build your TOD based on your database, actions, or knowledge base (as well as documents). We take TOD as a special kinds of `human inferface`, which means we define a TOD task as helping user obtain specific information (or knowledge) by `human language`, thus you must have the information you need to providing to user first.
* Intelligent analysis. With database, actions and knowledges, YADAN will assist you in the desgin of the structure of your own TODs. For example, with database linked, yadan will parse the dialogue schema (i.e. domains, slots, values...) automatically.
* Fast. ...

And yadan has blew limitations:
* Not flexible. You shouldn't take your academic exploration with YADAN. The best way is modifying the source code of some previous algorithms and using the unified evaluation. Yadan only aims to collect some classicial algorithms that is not garish, e.g. PIPELINE with transformer-style models, Nerual Pipeline with GPT-2, and totally end-to-end methods. A special model structure or a special training task or a special and not universal information flow will not be collected into yadan at first stages.
* No Chatbots. You shouldn't use yadan constructing a open-domain chatbot unless your chatbot is very similar to the format of a task-oriented dialogue. (These situation may happen when your chatbot needs a very concrete `topic` or rely on a unified knowledge base heavily.)


## Get Started

This is the dependency in cargo.tomal:
```toml
serde_json = "1.0"
serde="1.0"
serde_derive="1.0"
sqlx={version="0.5",features=["runtime-async-std-native-tls","sqlite","macros","tls"]}
sea-orm = { version = "0.5", features = [ "runtime-async-std-native-tls", "sqlx-sqlite", "macros" ], default-features = false}
anyhow = "1"
async-std = {version="1",features = ["attributes"]}
futures="0.3"
zip="0.5"
jieba-rs={version="0.6",features=["default-dict","tfidf","textrank"]}
word2vec="0.3.3"
ndarray="0.15.0"
linfa="0.5"
linfa-clustering="0.5"
linfa-nn="0.5"
num-traits="0.2"
ndarray-rand="0.2"
rand_isaac="0.3"
approx="0.5"
rand_core="0.6"
rust-bert="0.17.0"
rust_tokenizers="7.0.1"
tch="~0.6.0"
async-trait="0.1"
simple_input="0.4"
```
The most important thing is, Yadan needs `torch` (C++) and `rust-bert` (torch). you should choose a proper version of torch lib and add it to your path. reference: []()

And you this is an example for use yadan to make inference.

```rust
	// your pretrained model path
    let pretrained_path="/home/liangzi/backinference/augpt-bigdata/";
    let soloist_model:SOLOIST=SOLOIST::init(pretrained_path);

	// test language model generation
    let prefix_his:&str="I would like a taxi from Saint John's college to Pizza Hut Fen Ditton.";
    let result:String=soloist_model.generate(prefix_his);

	//running interaction.
    soloist_model.interact();
```
If you have a huggingface `transformers` pretrained language model already, just transfer it into a `rust-bert` format by running `cargo run --bin convert-utils your/source/model/path your/target/model/path`.

## Things To Do
Unfortunately recently I need to find a job because I am faced with the graduation, and the rate of progress may be very slow until Oct., 2022. Now I just public this repo and list the things to do as:

+ TO DO
  + training models with yadan;
  + the implemention the variants of neural pipeline (SimpleTOD, UBAR, and so on);
  + how to link the decided "intent" into APIs;
  + other datbase supports (now only sqlite)
  + structured knowledge based TOD
  + unstructured knowledge based TOD 
  + GUI interface;
  + online demo

+ IN PROGRESS

+ DONE
  + core desgin, including the basic data types and interface;
  + simple analyzer;
  + inference with SOLOIST

## Contact me
Feel free to give me a issue. And my wechat is: frostliangzi
