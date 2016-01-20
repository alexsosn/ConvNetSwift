//
//  AutoencoderTest.swift
//  ConvNetSwift
//
//  Created by Tanya on 12/4/15.
//  Copyright © 2015 OWL. All rights reserved.
//

import XCTest

class AutoencoderTest: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

    func testAutoencoderNet() {
        
        let input = InputLayerOpt(outSx: 423, outSy: 273, outDepth: 1)
        let fc1 = FullyConnLayerOpt(numNeurons: 50, activation: .Tanh)
        let fc2 = FullyConnLayerOpt(numNeurons: 50, activation: .Tanh)
        let fc3 = FullyConnLayerOpt(numNeurons: 2)
        let fc4 = FullyConnLayerOpt(numNeurons: 50, activation: .Tanh)
        let fc5 = FullyConnLayerOpt(numNeurons: 50, activation: .Tanh)
        let regression = RegressionLayerOpt(numNeurons: 423*273)
        let layerDefs: [LayerOptTypeProtocol] = [input, fc1, fc2, fc3, fc4, fc5, regression]

        let net = Net()
        net.makeLayers(layerDefs)
        
        var trainerOpts = TrainerOpt()
        trainerOpts.learningRate = 1
        trainerOpts.method = .adadelta
        trainerOpts.batchSize = 50
        trainerOpts.l2Decay = 0.001
        trainerOpts.l1Decay = 0.001

        let trainer = Trainer(net: net, options: trainerOpts)
        let image = UIImage(named: "Nyura.png")!
        var v = image.toVol()
        let res = trainer.train(x: &v, y: v.w)
        print(res)
    }
}
