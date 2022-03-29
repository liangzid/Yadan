// This file summerize all error types.

use std::fmt;
use std::io::{self,Read};

#[derive(Debug)]
pub struct YadanInferenceError{
    kind:String,
    message:String,
}

impl fmt::Display for YadanInferenceError {
    fn fmt(&self, f: &mut fmt::Formatter)-> fmt::Result{
	write!(f, "YADAN ERROR\n Kind: {}\n =>{}",self.kind, self.message)
    }
}

impl From<io::Error> for YadanInferenceError{
    fn from(error: io::Error)->Self{
	YadanInferenceError{kind: String::from("IO"),message:error.to_string()}
    }
}
