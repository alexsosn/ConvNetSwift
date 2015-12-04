//
//  JSTest.swift
//  ConvNetSwift
//
//  Created by Alex Sosnovshchenko on 11/4/15.
//  Copyright © 2015 OWL. All rights reserved.
//

import XCTest
import JavaScriptCore

class JSTest: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

    func testExample() {
        let context = JSContext()
        context.exceptionHandler = { context, exception in
            print("JS Error: \(exception)")
        }
        
        let path = NSBundle.mainBundle().pathForResource("convnet", ofType: "js")!
        context.evaluateScript(path)
//        context.evaluateScript("12.34a")
        context.evaluateScript("var num = 5 + 5")
        context.evaluateScript("var names = ['Grace', 'Ada', 'Margaret']")
        context.evaluateScript("var triple = function(value) { return value * 3 }")
        let tripleNum: JSValue = context.evaluateScript("triple(num)")
        XCTAssertEqual(tripleNum.toInt32(), 30)
        
        let tripleFunction = context.objectForKeyedSubscript("triple")
        let result = tripleFunction.callWithArguments([5])
        print("Five tripled: \(result.toInt32())")
    
    }
}
