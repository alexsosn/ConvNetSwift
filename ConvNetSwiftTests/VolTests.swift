//
//  VolTests.swift
//  ConvNetSwift
//
//  Created by Alex on 11/24/15.
//  Copyright © 2015 OWL. All rights reserved.
//

import XCTest

class VolTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

    func testRandomVol() {
        
        let vol = Vol(sx: 1,sy: 1,depth: 100)
        XCTAssertGreaterThan(vol.w.reduce(0, { (acc: Double, new: Double) -> Double in
            return acc+new
        }), 0)
        
        XCTAssertEqual(vol.dw.reduce(0, { (acc: Double, new: Double) -> Double in
            return acc+new
        }), 0)
    }
    
    func testPredefinedVol() {
        let vol = Vol(sx: 1,sy: 1,depth: 100,c: 5)
        XCTAssertEqual(vol.w.reduce(0, { (acc: Double, new: Double) -> Double in
            return acc+new
        }), 500)
        
        XCTAssertEqual(vol.dw.reduce(0, { (acc: Double, new: Double) -> Double in
            return acc+new
        }), 0)
    }
}
