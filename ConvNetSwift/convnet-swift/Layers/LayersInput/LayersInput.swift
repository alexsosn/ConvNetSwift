import Foundation

struct InputLayerOpt: LayerOutOptProtocol {
    var layerType: LayerType = .Input
    
    var outSx: Int
    var outSy: Int
    var outDepth: Int
    
    init(outSx: Int,
        outSy: Int,
        outDepth: Int) {
            self.outSx = outSx
            self.outSy = outSy
            self.outDepth = outDepth
    }
}

class InputLayer: InnerLayer {
    
    var outDepth: Int
    var outSx: Int
    var outSy: Int
    var layerType: LayerType
    
    var inAct: Vol?
    var outAct: Vol?
    
    init(opt: InputLayerOpt) {
        
        // required: depth
        self.outDepth = opt.outDepth ?? 0
        
        // optional: default these dimensions to 1
        self.outSx = opt.outSx ?? 1
        self.outSy = opt.outSy ?? 1
        
        // computed
        self.layerType = .Input
    }
    
    func forward(inout V: Vol, isTraining: Bool) -> Vol {
        self.inAct = V
        self.outAct = V
        return self.outAct! // simply identity function for now
    }
    
    func backward() -> () {}
    
    func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    func assignParamsAndGrads(paramsAndGrads: [ParamsAndGrads]) {
        
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        json["outDepth"] = self.outDepth
        json["outSx"] = self.outSx
        json["outSy"] = self.outSy
        json["layerType"] = self.layerType.rawValue
        return json
    }

//    func fromJSON(json: [String: AnyObject]) -> () {
//        self.outDepth = json["outDepth"]
//        self.outSx = json["outSx"]
//        self.outSy = json["outSy"]
//        self.layerType = json["layerType"]; 
//    }
}
