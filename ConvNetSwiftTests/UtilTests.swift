//
//  UtilTests.swift
//  ConvNetSwift
//
//  Created by Alex on 11/24/15.
//  Copyright © 2015 OWL. All rights reserved.
//

import XCTest

class UtilTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

    func testImageConversion() {
        
        guard let image = UIImage(named: "eel.jpg") else {
            print("error: no image found on provided path")
            XCTAssert(false)
            return
        }
        
        let vol = image.toVol()
        let newImage = vol.toImage()
        let newVol = newImage.toVol()
        
        XCTAssertEqual(vol.w.count, newVol.w.count)
        
        for i: Int in 0 ..< vol.w.count {
            let equals = vol.w[i] == newVol.w[i]
            XCTAssert(equals)
            if !equals {
                print(vol.w[i], i)
                print(newVol.w[i], i)
                break
            }
        }
    }
}
