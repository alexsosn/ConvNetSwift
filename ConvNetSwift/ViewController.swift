//
//  ViewController.swift
//  ConvNetSwift
//
//  Created by Alex Sosnovshchenko on 10/22/15.
//  Copyright © 2015 OWL. All rights reserved.
//

import UIKit

class ViewController: UIViewController {
    var net: Net?
    var trainer: Trainer?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        // Here's a minimum example of defining a 2-layer neural network and training it on a single data point:
        
        // species a 2-layer neural network with one hidden layer of 20 neurons
        // input layer declares size of input. here: 2-D data
        // ConvNetJS works on 3-Dimensional volumes (sx, sy, depth), but if you're not dealing with images
        // then the first two dimensions (sx, sy) will always be kept at size 1
        let input = InputLayerOpt(outSx: 1, outSy: 1, outDepth: 2)
        // declare 20 neurons, followed by ReLU (rectified linear unit non-linearity)
        let fc = FullyConnectedLayerOpt(numNeurons: 20, activation: .ReLU)
        // declare the linear classifier on top of the previous hidden layer
        let softmax = SoftmaxLayerOpt(numClasses: 2)
        
        let net = Net([input, fc, softmax])
        
        // forward a random data point through the network
        var x = Vol(array: [0.3, -0.5])
        let prob = net.forward(&x)
        
        // prob is a Vol. Vols have a field .w that stores the raw data, and .dw that stores gradients
        print("probability that x is class 0: \(prob.w[0])") // prints (for example) 0.602521202165062
        
        var traindef = TrainerOpt()
        traindef.learningRate = 0.01
        traindef.l2Decay = 0.001
        
        let trainer = Trainer(net: net, options: traindef)
        _ = trainer.train(x: &x, y: 0) // train the network, specifying that x is class zero
        
        let prob2 = net.forward(&x)
        print("probability that x is class 0: \(prob2.w[0])")
        // now prints (for example) 0.609982755733715, slightly higher than previous 0.602521202165062: the networks
        // weights have been adjusted by the Trainer to give a higher probability to
        // the class we trained the network with (zero)
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    
}
