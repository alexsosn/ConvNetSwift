//
//  SimpleNetTests.swift
//  ConvNetSwift
//
//  Created by Alex Sosnovshchenko on 11/3/15.
//  Copyright © 2015 OWL. All rights reserved.
//

import XCTest

class SimpleNetTests: XCTestCase {
    var net: Net?
    var trainer: Trainer?
    
    override func setUp() {
        super.setUp()
        srand48(time(nil))

        
        let input = InputLayerOpt(outSx: 1, outSy: 1, outDepth: 2)
        let fc1 = FullyConnectedLayerOpt(numNeurons: 50, activation: .Tanh)
        let fc2 = FullyConnectedLayerOpt(numNeurons: 40, activation: .Tanh)
        let softmax = SoftmaxLayerOpt(numClasses: 3)
        
        net = Net([input, fc1, fc2, softmax])
        
        var trainerOpts = TrainerOpt()
        trainerOpts.learningRate = 0.0001
        trainerOpts.momentum = 0.0
        trainerOpts.batchSize = 1
        trainerOpts.l2Decay = 0.0
        trainer = Trainer(net: net!, options: trainerOpts)
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    // should be possible to initialize
    func testInit() {
        
        // tanh are their own layers. Softmax gets its own fully connected layer.
        // this should all get desugared just fine.
        XCTAssertEqual(net!.layers.count, 7)
    }
    
    // should forward prop volumes to probabilities
    func testForward() {
        
        var x = Vol(array: [0.2, -0.3])
        let probabilityVolume = net!.forward(&x)
        
        XCTAssertEqual(probabilityVolume.w.count, 3)  // 3 classes output
        var w = probabilityVolume.w
        for i in 0 ..< 3 {
            XCTAssertGreaterThan(w[i], 0.0)
            XCTAssertLessThan(w[i], 1.0)
        }
        
        XCTAssertEqualWithAccuracy(w[0]+w[1]+w[2], 1.0, accuracy: 0.000000000001)
    }
    
    // should increase probabilities for ground truth class when trained
    func testTrain() {
        
        // lets test 100 random point and label settings
        // note that this should work since l2 and l1 regularization are off
        // an issue is that if step size is too high, this could technically fail...
        for _ in 0 ..< 100 {
            var x = Vol(array: [RandUtils.random_js() * 2 - 1, RandUtils.random_js() * 2 - 1])
            let pv = net!.forward(&x)
            let gti = Int(RandUtils.random_js() * 3)
            let trainRes = trainer!.train(x: &x, y: gti)
            print(trainRes)
            
            let pv2 = net!.forward(&x)
            XCTAssertGreaterThan(pv2.w[gti], pv.w[gti])
        }
    }
    
    // should compute correct gradient at data
    func testGrad() {
        
        // here we only test the gradient at data, but if this is
        // right then that's comforting, because it is a function
        // of all gradients above, for all layers.
        
        var x = Vol(array: [RandUtils.random_js() * 2.0 - 1.0, RandUtils.random_js() * 2.0 - 1.0])
        let gti = Int(RandUtils.random_js() * 3) // ground truth index
        let res = trainer!.train(x: &x, y: gti) // computes gradients at all layers, and at x
        
        print(res)
        
        let Δ = 0.000001
        
        for i: Int in 0 ..< x.w.count {

            // finite difference approximation
            
            let gradAnalytic = x.dw[i]
            
            let xold = x.w[i]
            x.w[i] += Δ
            let c0 = net!.getCostLoss(V: &x, y: gti)
            x.w[i] -= 2*Δ
            let c1 = net!.getCostLoss(V: &x, y: gti)
            x.w[i] = xold // reset
            
            let gradNumeric = (c0 - c1)/(2.0 * Δ)
            let relError = abs(gradAnalytic - gradNumeric)/abs(gradAnalytic + gradNumeric)
            print("\(i): numeric: \(gradNumeric), analytic: \(gradAnalytic) => rel error \(relError)")
            
            XCTAssertLessThan(relError, 1e-2)
            
        }
    }
    
}
