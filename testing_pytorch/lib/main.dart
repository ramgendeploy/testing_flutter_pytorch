import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:typed_data';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: const MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({Key key}) : super(key: key);

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  static const MethodChannel pytorchChannel =
      MethodChannel('com.pytorch_channel');

  @override
  void initState() {
    super.initState();
    _gettingModelFile().then((void value) => print('File Created Successfuly'));
  }

  String documentsPath;
  String prediction;

  Future<void> _gettingModelFile() async {
    final Directory directory = await getApplicationDocumentsDirectory();

    setState(() {
      documentsPath = directory.path;
    });

    final String resnet50 = join(directory.path, 'model.pt');
    final ByteData data = await rootBundle.load('assets/models/model.pt');

    final String segmodel = join(directory.path, 'seg_opt.pt');
    final ByteData segdata = await rootBundle.load('assets/models/seg_opt.pt');

    final List<int> bytes =
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    final List<int> segbytes = segdata.buffer
        .asUint8List(segdata.offsetInBytes, segdata.lengthInBytes);

    if (!File(resnet50).existsSync()) {
      await File(resnet50).writeAsBytes(bytes);
    }
    if (!File(segmodel).existsSync()) {
      await File(segmodel).writeAsBytes(segbytes);
    }
  }

  Future<void> _getPrediction() async {
    final ByteData imageData = await rootBundle.load('assets/animal.jpg');
    try {
      final String result = await pytorchChannel.invokeMethod(
        'predict_image',
        <String, dynamic>{
          'model_path': '$documentsPath/model.pt',
          'image_data': imageData.buffer
              .asUint8List(imageData.offsetInBytes, imageData.lengthInBytes),
          'data_offset': imageData.offsetInBytes,
          'data_length': imageData.lengthInBytes
        },
      );
      setState(() {
        prediction = result;
      });
    } on PlatformException catch (e) {
      print(e);
    }
  }

  Future<void> _getSegmentation() async {
    final ByteData imageData = await rootBundle.load('assets/animal.jpg');
    try {
      final Float64List result = await pytorchChannel.invokeMethod(
        'segment_image',
        <String, dynamic>{
          'model_path': '$documentsPath/seg_opt.pt',
          'image_data': imageData.buffer
              .asUint8List(imageData.offsetInBytes, imageData.lengthInBytes),
          'data_offset': imageData.offsetInBytes,
          'data_length': imageData.lengthInBytes
        },
      );
      print(result);

      // setState(() {
      //   prediction = result;
      // });
    } on PlatformException catch (e) {
      print(e);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Pytorch Mobile'),
      ),
      body: Center(
        child: Column(
          children: <Widget>[
            Text(documentsPath ?? ''),
            Stack(
              children: <Widget>[
                Align(
                    alignment: Alignment.center,
                    child: SizedBox(
                        width: 300, child: Image.asset('assets/animal.jpg'))),
                Align(
                  alignment: Alignment.bottomCenter,
                  child: Text(
                    (prediction ?? '').toUpperCase(),
                    style: const TextStyle(fontSize: 25),
                  ),
                ),
              ],
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: <Widget>[
                RaisedButton(
                  child: const Text('Segment Image'),
                  onPressed: _getSegmentation,
                ),
                RaisedButton(
                  child: const Text('Classify Image'),
                  onPressed: _getPrediction,
                )
              ],
            )
          ],
        ),
      ),
      // floatingActionButton: FloatingActionButton(
      //   onPressed: _getPrediction,
      //   tooltip: 'Predict Image',
      //   child: const Icon(Icons.add),
      // ),
    );
  }
}
