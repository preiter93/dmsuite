extern crate ndarray;
#[allow(unused_imports)]
use ndarray::*;

mod common;
mod chebdif;
mod fourdif;
pub use crate::common::*;

pub const PI: f64 = 3.14159265358979323846264338327950288f64;

#[allow(dead_code)]
#[allow(unused_variables)]
fn main() {
    println!("Hello, world!");
    let n = 3;
    // Construct Dmatrices
    let (x,d1) = chebdif::chebdif(n,1);
    let (x,d2) = chebdif::chebdif(n,2);
    let (x,d1) = fourdif::fourdif(n,1);
    let (x,d2) = fourdif::fourdif(n,2);
    // Differentiate
    let f = (&x).mapv(f64::sin);
    let df_exp = (&x).mapv(f64::cos);
    let ddf_exp = (&x).mapv(f64::sin);
    let d3 = d1.dot(&d2);
    
    println!("Finished!");
}

