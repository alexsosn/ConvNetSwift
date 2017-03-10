
public protocol LayerOptTypeProtocol {
    var layerType: LayerType {get}
}

public protocol LayerOutOptProtocol: LayerOptTypeProtocol {
    var outSx: Int {get set}
    var outSy: Int {get set}
    var outDepth: Int {get set}
}

public protocol LayerInOptProtocol: LayerOptTypeProtocol {
    var inSx: Int {get set}
    var inSy: Int {get set}
    var inDepth: Int {get set}
}

public protocol LayerOptActivationProtocol {
    var activation: ActivationType {get set}
}

public protocol DropProbProtocol {
    var dropProb: Double? {get set}
}
