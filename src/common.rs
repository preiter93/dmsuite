extern crate ndarray;
use ndarray::*;

/// Equivalent to np.tile()
pub fn tile<S>(a: &ArrayBase<S,Ix1>) ->  Array2<f64>
where
	S: Data<Elem=f64>
{
	let n = a.len();
	let mut rv = Array2::<f64>::zeros((n,n));

	for mut row in rv.axis_iter_mut(Axis(0)) {
        row += a;
    }
    return rv
}

/// Set diagonals of matrix to fixed value
pub fn set_diag(arr: &mut Array2<f64>, val: f64) {
	assert!(arr.shape()[0] == arr.shape()[1]  );
	for a_ii in arr.diag_mut() {
        *a_ii = val;
    }
}

pub fn set_diag_from_arr(arr: &mut Array2<f64>, val: &Array1<f64>) {
	assert!(arr.shape()[0] == arr.shape()[1]  );
	assert!(arr.shape()[0] == val.len() );
	for (i,a_ii) in arr.diag_mut().iter_mut().enumerate() {
        *a_ii = val[i];
    }
}

/// Constructs a toeplitz matrix from 1 array
/// See also fn toeplitz
pub fn toeplitz1<S>(a: &ArrayBase<S,Ix1>) ->  Array2<f64>
where
	S: Data<Elem=f64>
{
	let n = a.len();
	let mut rv = Array2::<f64>::zeros((n,n));
 	// Lower Diagonal
 	for i in 0..n{
 		for (ii,j) in (0..i).rev().enumerate(){
 			rv[[i,j]] = a[ii+1];
 		}
 	}
 	// Upper Diagonal
 	rv = &rv+&rv.t();
 	// Main Diagonal
 	set_diag(&mut rv, a[0]);
    return rv
}

/// Constructs a toeplitz matrix from 2 arrays l & u
/// l and u must have same size and same first element
pub fn toeplitz<S>(l: &ArrayBase<S,Ix1>,u: &ArrayBase<S,Ix1>) ->  Array2<f64>
where
    S: Data<Elem=f64>
{
    // Check
    assert!(l.len()==u.len());
    assert!(l[0]==u[0]);

    let n = l.len();
    let mut rv = Array2::<f64>::zeros((n,n));
    // Lower Diagonal
    for i in 0..n{
        for (ii,j) in (0..i).rev().enumerate(){
            rv[[i,j]] = l[ii+1];
        }
    }
    // Upper Diagonal
    for i in 0..n-1{
        for (ii,j) in (i+1..n).enumerate(){
            rv[[i,j]] = u[ii+1];
        }
    }
    // Main Diagonal
    set_diag(&mut rv, l[0]);
    return rv
}

pub fn flipud<S>(aa: &ArrayBase<S,Ix2>) ->  Array2<f64>
where
	S: Data<Elem=f64>
{
	let mut rv = Array2::zeros(aa.raw_dim());
	rv.slice_mut(s![..,..]).assign(&aa.slice(s![..;-1,..]));
	return rv
}

pub fn fliplr<S>(aa: &ArrayBase<S,Ix2>) ->  Array2<f64>
where
	S: Data<Elem=f64>
{
	let mut rv = Array2::zeros(aa.raw_dim());
	rv.slice_mut(s![..,..]).assign(&aa.slice(s![..,..;-1]));
	return rv
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    //use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_flipud (){
    	// Init
    	let aa =arr2(&[[1.,2.,3.], [4.,5.,6.], [7.,8.,9.]]);
        let bb =arr2(&[[7.,8.,9.], [4.,5.,6.], [1.,2.,3.]]);
        // Run
        let cc = flipud(&aa);
        // Check
        assert_eq!(cc, bb);
    }

    #[test]
    fn test_flipud_view (){
    	// Init
    	let aa =arr2(&[[1.,2.,3.], [4.,5.,6.], [7.,8.,9.]]);
        let bb =arr2(&[[7.,8.,9.], [4.,5.,6.], [1.,2.,3.]]);
        // Run
        let cc = flipud(&aa.view());
        // Check
        assert_eq!(cc, bb);
    }

    #[test]
    fn test_fliplr (){
    	// Init
    	let aa =arr2(&[[1.,2.,3.], [4.,5.,6.], [7.,8.,9.]]);
        let bb =arr2(&[[3.,2.,1.], [6.,5.,4.], [9.,8.,7.]]);
        // Run
        let cc = fliplr(&aa);
        // Check
        assert_eq!(cc, bb);
    }

    #[test]
    fn test_toeplitz(){
    	// Init
    	let l = Array::range(0., 3., 1.);
    	let u =-Array::range(0., 3., 1.);
        let t_exp =arr2(&[[0.,-1.0,-2.],[1.,0.,-1.],[2.,1.,0.]]);
        // Run
        let t = toeplitz(&l,&u);
        // Check
        assert_eq!(t, t_exp);
    }

    #[test]
    #[should_panic]
    /// l and u must be same size
    fn test_toeplitz_fail(){
    	// Init
    	let l = Array::range(0., 3., 1.);
    	let u =-Array::range(1., 3., 1.); 
        // Run
        let _ = toeplitz(&l,&u);
    }

	#[test]
    #[should_panic]
    /// l and u must have same first element
    fn test_toeplitz_fail2(){
    	// Init
    	let l = Array::range(0., 3., 1.);
    	let mut u =-Array::range(0., 3., 1.); 
    	u[0] = 99.;
        // Run
        let _ = toeplitz(&l,&u);
    }
}


    // let test =arr2(&[[1.,2.,3.], [4.,5.,6.], [7.,8.,9.]]);
    
    // let k = Array::range(0., 3.0, 1.0);
    // let t = toeplitz(&k,&(-1.*k.clone() ));

// /// Roll along on-dimensional array. 
// /// No suppot for negative roll yet.
// fn roll(arr: &Array1<f64>, roll: usize) -> Array1<f64> {
// 	assert!(arr.len() > roll );
// 	let n = arr.len();
// 	let mut rv = Array1::<f64>::zeros(n);
// 	for (i, a) in rv.slice_mut(s![roll..]).iter_mut().enumerate(){
// 		*a = arr[i];
// 	}
// 	let offset = n-roll;
// 	for (i, a) in rv.slice_mut(s![..roll]).iter_mut().enumerate(){
// 		*a = arr[offset+i];
// 	}
// 	return rv
// }

