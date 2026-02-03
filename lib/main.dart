import 'dart:typed_data';
import 'package:flutter/foundation.dart'; 
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

late List<CameraDescription> _cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  _cameras = await availableCameras();
  runApp(const MaterialApp(
    home: FaceApp(),
    debugShowCheckedModeBanner: false,
  ));
}

class FaceApp extends StatefulWidget {
  const FaceApp({super.key});

  @override
  State<FaceApp> createState() => _FaceAppState();
}

class _FaceAppState extends State<FaceApp> {
  CameraController? controller;
  Interpreter? interpreter;
  int _cameraIndex = 0; // Camera index track karnyasathi
  
  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(
      performanceMode: FaceDetectorMode.accurate, 
    ),
  );

  String status = "Initializing...";
  bool isProcessing = false;

  @override
  void initState() {
    super.initState();
    // Default front camera shodha, nasel tar pahila camera ghya
    _cameraIndex = _cameras.indexWhere((c) => c.lensDirection == CameraLensDirection.front);
    if (_cameraIndex == -1) _cameraIndex = 0;
    
    initApp(_cameraIndex);
  }

  Future<void> initApp(int index) async {
    try {
      if (controller != null) await controller!.dispose();

      controller = CameraController(
        _cameras[index],
        ResolutionPreset.high, 
        enableAudio: false,
      );

      await controller!.initialize();
      interpreter ??= await Interpreter.fromAsset('assets/face_model.tflite');
      
      if (mounted) {
        setState(() => status = "Camera Ready");
      }
    } catch (e) {
      setState(() => status = "Error: $e");
    }
  }

  // Camera switch karnyache function
  void toggleCamera() {
    if (_cameras.length < 2) return;
    _cameraIndex = (_cameraIndex + 1) % _cameras.length;
    initApp(_cameraIndex);
  }

  Future<void> captureAndCheck() async {
    if (isProcessing || controller == null) return;
    
    setState(() {
      isProcessing = true;
      status = "Analyzing Face...";
    });

    try {
      XFile file = await controller!.takePicture();
      Uint8List bytes = await file.readAsBytes();
      
      final inputImage = InputImage.fromFilePath(file.path);
      final faces = await _faceDetector.processImage(inputImage);
      
      if (faces.isEmpty) {
        setState(() {
          status = "❌ Face not detected! Try closer.";
          isProcessing = false;
        });
        return;
      }

      img.Image? original = img.decodeImage(bytes);
      if (original == null) return;

      // Check kara ki camera front aahe ki back
      bool isFront = _cameras[_cameraIndex].lensDirection == CameraLensDirection.front;
      
      // Front camera sathi 270 degree rotation lagte, back sathi garaj naste
      img.Image processedImg = isFront ? img.copyRotate(original, angle: 270) : original;

      final face = faces.first;
      final rect = face.boundingBox;

      img.Image faceCrop = img.copyCrop(
        processedImg,
        x: rect.left.toInt().clamp(0, processedImg.width),
        y: rect.top.toInt().clamp(0, processedImg.height),
        width: rect.width.toInt().clamp(10, processedImg.width),
        height: rect.height.toInt().clamp(10, processedImg.height),
      );

      img.Image resized = img.copyResize(faceCrop, width: 96, height: 96);

      var input = List.generate(1, (_) => List.generate(96, (y) => List.generate(96, (x) {
        final p = resized.getPixel(x, y);
        return [
          (p.b - 127.5) / 127.5, 
          (p.g - 127.5) / 127.5, 
          (p.r - 127.5) / 127.5
        ]; 
      })));

      var output = List.generate(1, (_) => List.filled(1, 0.0));
      interpreter!.run(input, output);

      double score = output[0][0];
      debugPrint("RAW SCORE FROM MODEL: $score");

      setState(() {
        if (score < 0.3) {
          status = "✅ REAL FACE (${(score * 100).toStringAsFixed(1)}%)";
        } else {
          status = "❌ SPOOF / PHOTO (${(score * 100).toStringAsFixed(1)}%)";
        }
        isProcessing = false;
      });

    } catch (e) {
      setState(() {
        status = "Error: $e";
        isProcessing = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (controller == null || !controller!.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    return Scaffold(
      appBar: AppBar(
        title: const Text("AI Face Detector"), 
        backgroundColor: Colors.blueAccent,
        actions: [
          // Camera Switch Button
          IconButton(
            icon: const Icon(Icons.flip_camera_ios, color: Colors.white),
            onPressed: isProcessing ? null : toggleCamera,
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(child: CameraPreview(controller!)),
          Container(
            padding: const EdgeInsets.all(25),
            decoration: const BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
            ),
            child: Column(
              children: [
                Text(
                  status,
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                    color: status.contains("✅") ? Colors.green : Colors.red,
                  ),
                ),
                const SizedBox(height: 20),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: isProcessing ? null : captureAndCheck,
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.all(18),
                      backgroundColor: Colors.blueAccent,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
                    ),
                    child: isProcessing 
                      ? const CircularProgressIndicator(color: Colors.white) 
                      : const Text("CHECK NOW", style: TextStyle(fontSize: 18, color: Colors.white)),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    controller?.dispose();
    interpreter?.close();
    _faceDetector.close();
    super.dispose();
  }
}