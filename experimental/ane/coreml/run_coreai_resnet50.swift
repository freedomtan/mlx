// Load and run a Core AI .aimodel for ResNet50, sustained, so it can be
// measured the same way as the raw-MPSGraph CompiledANE path and the Core ML
// path: ANECompilerService log capture for procedure count + rootless
// powermetrics sampling for ANE engagement.
//
// Build:
//   swiftc -O -parse-as-library run_coreai_resnet50.swift -o /tmp/run_coreai_resnet50
// Run:
//   /tmp/run_coreai_resnet50 <path-to-resnet50.aimodel> <seconds>

import CoreAI
import Foundation

@main
struct RunCoreAI {
    static func main() async throws {
        let args = CommandLine.arguments
        guard args.count > 1 else {
            print("usage: \(args[0]) <model path> [seconds]")
            exit(1)
        }
        let modelURL = URL(fileURLWithPath: args[1])
        let seconds = args.count > 2 ? Double(args[2]) ?? 5.0 : 5.0

        print("[run_coreai] available compute units: \(ComputeUnitKind.availableKinds)")

        let options = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)

        print("[run_coreai] loading+specializing \(modelURL.path)...")
        let tLoad0 = Date()
        let model = try await AIModel(contentsOf: modelURL, options: options)
        let tLoad1 = Date()
        print("[run_coreai] loaded in \(Int((tLoad1.timeIntervalSince(tLoad0)) * 1000))ms")
        print("[run_coreai] function names: \(model.functionNames)")

        guard let fnName = model.functionNames.first,
              let fn = try model.loadFunction(named: fnName) else {
            print("no function found")
            exit(1)
        }
        guard let desc = model.functionDescriptor(for: fnName) else {
            print("no descriptor")
            exit(1)
        }
        print("[run_coreai] function descriptor: \(desc)")

        let shape: [Int] = [1, 3, 224, 224]
        var floatData = [Float16](repeating: 0, count: shape.reduce(1, *))
        for i in 0..<floatData.count {
            floatData[i] = Float16((i % 997) - 498) / 498.0
        }
        let input = NDArray(scalars: floatData, shape: shape)

        func runAndMaterialize() async throws -> Float16 {
            var outputs = try await fn.run(
                inputs: ["x": input],
                states: InferenceFunction.MutableViews(),
                outputViews: InferenceFunction.MutableViews()
            )
            guard let value = outputs.remove("relu_48"),
                  let arr = value.ndArray else {
                fatalError("no output ndArray")
            }
            let view = arr.view(as: Float16.self)
            let acc: Float = view.withUnsafePointer { pointer, shape, _ in
                var total: Float = 0
                var count = 1
                for i in 0..<shape.count { count *= shape[i] }
                for i in 0..<count { total += Float(pointer[i]) }
                return total
            }
            return Float16(acc)
        }

        print("[run_coreai] warm-up inference...")
        var sink: Float16 = try await runAndMaterialize()
        print("[run_coreai] warm-up ok, output sum: \(sink)")

        print("[run_coreai] sustained starting now, running for \(seconds)s")
        let t0 = Date()
        var iters = 0
        while Date().timeIntervalSince(t0) < seconds {
            sink = try await runAndMaterialize()
            iters += 1
        }
        let dt = Date().timeIntervalSince(t0)
        print("[run_coreai] sustained done: \(iters) iters in \(Int(dt * 1000))ms (\(dt * 1000.0 / Double(iters))ms/iter), sink=\(sink)")
    }
}
