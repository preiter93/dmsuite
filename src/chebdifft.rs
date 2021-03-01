extern crate ndarray;
use ndarray::*;
use rustfft::{FftPlanner, num_complex::Complex};
use realfft::RealFftPlanner;

// Perform chebyshev spectral differentiation via real to complex FFT 
// Input:
// f:	    Vector of type Vec<f64> containing function valuesat the Chebyshev points
//          x(k) = cos((k)*pi/(N-1)), k = 0...N-1.
// der: 	Derivative order of type usize
// Output:
// outdata: Vector containing the approximate der'th derivative
//
// Note: 
// This version uses the faster real to complex fft
pub fn chebdifft(f: &mut Array1<f64>, der: usize) -> Array1<f64> {
	let n = f.len(); 
	//// Setup real to complex FFT
	let length = 2*n-2; 
	// Planner
	let mut real_planner = RealFftPlanner::<f64>::new(); 
	// create a FFT
	let r2c = real_planner.plan_fft_forward(length); 
	// output vector
	let mut spectrum = r2c.make_output_vec();

	// Extend and compute fft
	let a0 = concatenate(Axis(0),
		&[f.view(),(f).slice(s![1..n-1;-1])]).unwrap();
	// Forward transform of real input vector
	let mut indata: Vec<f64> = a0.to_vec().clone();
	r2c.process(&mut indata, &mut spectrum).unwrap();

	// Unpack Real part, a0 contains Chebychev coefficient of f
	let mut a0: Array1<f64> = Array::from(spectrum[0..n]
		.iter().map(|x| x.re).collect::<Vec<f64>>() );
	let b0: Array1<f64> = concatenate(Axis(0), // chain together
		&[ arr1(&[0.5]).view(),Array::ones(n-2).view(),arr1(&[0.5]).view()]).unwrap()
		/(n as f64 -1.0);
	a0 *= &b0;
	
	// Recursion formula for computing coefficients
	let mut a:Array2<f64> = Array::zeros( (n,der+1));
	a.slice_mut(s![..,0]).assign(&a0);
	for ell in 1..der+1{
		a[[n-ell-1,ell]] = 2.*(n-ell) as f64*a[[n-ell,ell-1]];
		for k in (1..n-ell-1).rev(){
			a[[k,ell]]=a[[k+2,ell]]+2.*(k+1) as f64*a[[k+1,ell-1]];
		a[[0,ell]] = a[[1,ell-1]] + a[[2,ell]]/2.0;
		}
	}
	// println!("{:?}", a);
	// Transform back into physical space
	let b1 = arr1(&[2.*&a[[0,  der]]]);
	let b3 = arr1(&[2.*&a[[n-1,der]]]);
	let b2 = a.slice(s![1..n-1,   der]);
	let b4 = a.slice(s![1..n-1;-1,der]);
	let back = concatenate(Axis(0), // chain together
		&[ b1.view(),b2,b3.view(),b4] ).unwrap();

	// Forward transform of real input vector
	let mut indata: Vec<f64> = back.to_vec().clone();
	r2c.process(&mut indata, &mut spectrum).unwrap();
	// Unpack Real part
	return Array::from(spectrum.iter().map(|x| 0.5*x.re).collect::<Vec<f64>>() )
}


// Perform chebyshev spectral differentiation via FFT (complex to complex)
// Input:
// f:	    Vector of type Vec<f64> containing function valuesat the Chebyshev points
//          x(k) = cos((k)*pi/(N-1)), k = 0...N-1.
// der: 	Derivative order of type usize
// Output:
// outdata: Vector containing the approximate der'th derivative
//
// Note: 
// This version uses the c2c fft to transform to chebyshev space, 
// which does redundant operations for real valued input (better: fn chebdifft)
#[allow(dead_code)]
pub fn chebdifft_c2c(f: &mut Array1<f64>, der: usize) -> Array1<f64> {
	let n = f.len(); 
	// Setup FFT
	let length = 2*n-2;
	let mut planner = FftPlanner::<f64>::new();
	let fft = planner.plan_fft_forward(length);
	let mut buffer = vec![Complex{ re: 0.0f64, im: 0.0f64 }; length];
	// Extend and compute fft
	let a0 = concatenate(Axis(0),
		&[f.view(),(f).slice(s![1..n-1;-1])]).unwrap();
	// Transfer data to buffer
	for (i,val) in &mut buffer.iter_mut().enumerate() { 
		val.re = a0[i];
	}
	// Perform forward fft
	fft.process(&mut buffer);
	// Unpack Real part, a0 contains Chebychev coefficient of f
	let mut a0: Array1<f64> = Array::from(
		buffer[0..n].iter().map(|x| x.re).collect::<Vec<f64>>() );
	let b0: Array1<f64> = concatenate(Axis(0), // chain together
		&[ arr1(&[0.5]).view(),Array::ones(n-2).view(),arr1(&[0.5]).view()]).unwrap()
		/(n as f64 -1.0);
	a0 *= &b0;

	// Recursion formula for computing coefficients
	let mut a:Array2<f64> = Array::zeros( (n,der+1));
	a.slice_mut(s![..,0]).assign(&a0);
	for ell in 1..der+1{
		a[[n-ell-1,ell]] = 2.*(n-ell) as f64*a[[n-ell,ell-1]];
		for k in (1..n-ell-1).rev(){
			a[[k,ell]]=a[[k+2,ell]]+2.*(k+1) as f64*a[[k+1,ell-1]];
		a[[0,ell]] = a[[1,ell-1]] + a[[2,ell]]/2.0;
		}
	}
	// println!("{:?}", a);
	// Transform back into physical space
	let b1 = arr1(&[2.*&a[[0,  der]]]);
	let b3 = arr1(&[2.*&a[[n-1,der]]]);
	let b2 = a.slice(s![1..n-1,   der]);
	let b4 = a.slice(s![1..n-1;-1,der]);
	let back = concatenate(Axis(0), // chain together
		&[ b1.view(),b2,b3.view(),b4] ).unwrap();
	// Transfer data to buffer
	for (i,val) in &mut buffer.iter_mut().enumerate() { 
		val.re = back[i];
	}
	// Perform fft
	fft.process(&mut buffer);
	// Unpack Real part
	return Array::from(buffer[0..n]
			.iter().map(|x| 0.5*x.re).collect::<Vec<f64>>() )
}


#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    extern crate time_test;
    use time_test::*;
    pub const PI: f64 = 3.14159265358979323846264338327950288f64;
    const N: usize = 10000;
    const TOL: f64 = 1e-3f64;
  
    #[test]
    fn test_chebdifft_c2c (){
    	time_test!();
        // Init
	   	let n = N;
    	let k = Array::range(0., n as f64, 1.0);
    	let x = (PI*k/(n as f64 - 1.0)).mapv(f64::cos); //Chebyshev nodes
    	let f = (&x*PI).mapv(f64::sin);
    	let d1f = PI*(&x*PI).mapv(f64::cos);
    	let d2f = -1.*PI*PI*(&x*PI).mapv(f64::sin);
    	// Run 
    	let mut indata: Array1<f64> = f.clone();
    	let d1f_cheb = chebdifft_c2c(&mut indata,1); // First derivative
    	let mut indata: Array1<f64> = f.clone();
    	let d2f_cheb = chebdifft_c2c(&mut indata,2); // First derivative
    	{time_test!("CHEBDIFFT Time");}
        // Check element-wise (maybe replace by norm)
        let tol = TOL;
        for (af, bf) in d1f.iter().zip(d1f_cheb.iter()) {
            assert_approx_eq!(af, bf, tol); // Check correctness
        }
        let d2f = d2f.slice(s![n/20..n-n/20]); // Cut off edge nodes for check
        let d2f_cheb = d2f_cheb.slice(s![n/20..n-n/20]);
        for (af, bf) in d2f.iter().zip(d2f_cheb.iter()) {
            assert_approx_eq!(af, bf, tol*1000.); // Check correctness
        }
    }

    #[test]
    fn test_chebdifft_r2c (){
    	time_test!();
        // Init
	   	let n = N;
    	let k = Array::range(0., n as f64, 1.0);
    	let x = (PI*k/(n as f64 - 1.0)).mapv(f64::cos); //Chebyshev nodes
    	let f = (&x*PI).mapv(f64::sin);
    	let d1f = PI*(&x*PI).mapv(f64::cos);
    	let d2f = -1.*PI*PI*(&x*PI).mapv(f64::sin);
    	// Run 
    	let mut indata: Array1<f64> = f.clone();
    	let d1f_cheb = chebdifft(&mut indata,1); // First derivative
    	let mut indata: Array1<f64> = f.clone();
    	let d2f_cheb = chebdifft(&mut indata,2); // First derivative
    	{time_test!("CHEBDIFFT Time");}
        // Check element-wise (maybe replace by norm)
        let tol = TOL;
        for (af, bf) in d1f.iter().zip(d1f_cheb.iter()) {
            assert_approx_eq!(af, bf, tol); // Check correctness
        }
        let d2f = d2f.slice(s![n/20..n-n/20]); // Cut off edge nodes for check
        let d2f_cheb = d2f_cheb.slice(s![n/20..n-n/20]);
        for (af, bf) in d2f.iter().zip(d2f_cheb.iter()) {
            assert_approx_eq!(af, bf, tol*1000.); // Check correctness
        }
    }

}