extern crate galerkin;
extern crate rand;

use galerkin::distmesh::distmesh_2d::*;
use galerkin::plot::glium::*;
use std::sync::mpsc::Sender;
use std::thread;
use std::time::Duration;

fn main() {
    let mesh = ellipse();
    let mesh_clone = ellipse();
    let f = move |sender: Sender<Vec<f64>>| loop {
        let values = mesh_clone
            .points
            .iter()
            .map(|_| rand::random::<f64>())
            .collect();
        sender.send(values).unwrap();
        thread::sleep(Duration::from_millis(300));
    };
    run_inside_plot(mesh, f);
}
