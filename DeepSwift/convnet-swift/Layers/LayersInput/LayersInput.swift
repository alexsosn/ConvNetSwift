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
        outDepth = opt.outDepth
        
        // optional: default these dimensions to 1
        outSx = opt.outSx
        outSy = opt.outSy
        
        // computed
        layerType = .Input
    }
    
    func forward(_ V: inout Vol, isTraining: Bool) -> Vol {
        inAct = V
        outAct = V
        return outAct! // simply identity function for now
    }
    
    func backward() -> () {}
    
    func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) {
        
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        json["outDepth"] = outDepth as AnyObject?
        json["outSx"] = outSx as AnyObject?
        json["outSy"] = outSy as AnyObject?
        json["layerType"] = layerType.rawValue as AnyObject?
        return json
    }

//    func fromJSON(json: [String: AnyObject]) -> () {
//        outDepth = json["outDepth"]
//        outSx = json["outSx"]
//        outSy = json["outSy"]
//        layerType = json["layerType"]; 
//    }
}
