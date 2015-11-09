import Foundation

struct InputLayerOpt: LayerOutOptProtocol {
    var layer_type: LayerType = .input
    
    var out_sx: Int
    var out_sy: Int
    var out_depth: Int
    
    init(out_sx: Int,
        out_sy: Int,
        out_depth: Int) {
            self.out_sx = out_sx
            self.out_sy = out_sy
            self.out_depth = out_depth
    }
}

class InputLayer: InnerLayer {
    
    var out_depth: Int
    var out_sx: Int
    var out_sy: Int
    var layer_type: LayerType
    
    var in_act: Vol?
    var out_act: Vol?
    
    init(opt: InputLayerOpt){
        
        // required: depth
        self.out_depth = opt.out_depth ?? 0
        
        // optional: default these dimensions to 1
        self.out_sx = opt.out_sx ?? 1
        self.out_sy = opt.out_sy ?? 1
        
        // computed
        self.layer_type = .input
    }
    
    func forward(inout V: Vol, is_training: Bool) -> Vol {
        self.in_act = V
        self.out_act = V
        return self.out_act! // simply identity function for now
    }
    
    func backward() -> () { }
    
    func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    func assignParamsAndGrads(paramsAndGrads: [ParamsAndGrads]) {
        
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        json["out_depth"] = self.out_depth
        json["out_sx"] = self.out_sx
        json["out_sy"] = self.out_sy
        json["layer_type"] = self.layer_type.rawValue
        return json
    }
//
//    func fromJSON(json: [String: AnyObject]) -> () {
//        self.out_depth = json["out_depth"]
//        self.out_sx = json["out_sx"]
//        self.out_sy = json["out_sy"]
//        self.layer_type = json["layer_type"]; 
//    }
}

