//
//  FCTests.swift
//  ConvNetSwift
//
//  Created by Alex on 1/20/16.
//  Copyright © 2016 OWL. All rights reserved.
//

import XCTest

class FCTests: XCTestCase {

    func testRandomInitialisation() {
        
        let input = InputLayerOpt(outSx: 1, outSy: 1, outDepth: 2)
        let softmax = SoftmaxLayerOpt(numClasses: 10)
        let net = Net([input, softmax])
        
        XCTAssertEqual(net.layers.count, 3, "Additional layer should be added between input and softmax.")
        XCTAssert(net.layers[1] is FullyConnectedLayer, "FC layer should be added between input and softmax.")
        let fc0 = net.layers[1] as! FullyConnectedLayer
        
        let filters = fc0.filters.flatMap{ $0.w }
        
        print(filters)
        
        for i in 0 ..< filters.count {
            for j in i+1 ..< filters.count {
                XCTAssertNotEqual(filters[i], filters[j], "All weights should be random and not equal to each other.")
            }
        }
    }
}
