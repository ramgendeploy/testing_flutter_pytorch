package com.ramgendeploy.testingpytorch.testing_pytorch;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import androidx.annotation.NonNull;
import com.ramgendeploy.testingpytorch.testing_pytorch.Constants;
import io.flutter.embedding.android.FlutterActivity;
import io.flutter.embedding.engine.FlutterEngine;
import io.flutter.plugin.common.MethodChannel;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Map;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

public class MainActivity extends FlutterActivity {
  private static final String CHANNEL = "samples.flutter.dev/battery";
  private static final String PYTORCH_CHANNEL = "com.pytorch_channel";

  @Override
  public void configureFlutterEngine(@NonNull FlutterEngine flutterEngine) {
    super.configureFlutterEngine(flutterEngine);

    new MethodChannel(
      flutterEngine.getDartExecutor().getBinaryMessenger(),
      PYTORCH_CHANNEL
    )
    .setMethodCallHandler(
        (call, result) -> {
          switch (call.method) {
            case "predict_image":
              Bitmap bitmap = null;
              Module module = null;
              try {
                String absPath = call.argument("model_path");
                int boffset = call.argument("data_offset");
                int blenght = call.argument("data_length");
                // bitmap = BitmapFactory.decodeStream(getAssets().open("image.jpg"));
                byte[] byteStream = call.argument("image_data");
                bitmap =
                  BitmapFactory.decodeByteArray(byteStream, boffset, blenght);

                Log.i("Pytorch: Main activity", absPath);

                module = Module.load(absPath);
              } catch (Exception e) {
                Log.e("Pytorch: Main activity", "Error reading", e);
                finish();
              }
              Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                bitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB
              );

              final Tensor oTensor = module
                .forward(IValue.from(inputTensor))
                .toTensor();

              float[] scores = oTensor.getDataAsFloatArray();

              float maxScore = -Float.MAX_VALUE;
              int maxScoreIdx = -1;
              for (int i = 0; i < scores.length; i++) {
                if (scores[i] > maxScore) {
                  maxScore = scores[i];
                  maxScoreIdx = i;
                }
              }
              String className = Constants.IMAGENET_CLASSES[maxScoreIdx];
              result.success(className);

              break;
            case "segment_image":
              // bitmap = Bitmap.createScaledBitmap(bitmap, width, height, false);
              Bitmap segbitmap = null;
              Module segmodule = null;
              try {
                String absPath = call.argument("model_path");
                int boffset = call.argument("data_offset");
                int blenght = call.argument("data_length");
                // bitmap = BitmapFactory.decodeStream(getAssets().open("image.jpg"));
                byte[] byteStream = call.argument("image_data");
                segbitmap =
                  BitmapFactory.decodeByteArray(byteStream, boffset, blenght);

                Log.i("Pytorch: Main activity", absPath);

                segmodule = Module.load(absPath);
              } catch (Exception e) {
                Log.e("Pytorch: Main activity", "Error reading", e);
                finish();
              }
              Tensor seginputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                segbitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB
              );

              IValue segoTensor = segmodule.forward(
                IValue.from(seginputTensor)
              );
              Map<String, IValue> dict = segoTensor.toDictStringKey();
              // Log.i("Pytorch", dict);
              for (String keys : dict.keySet()) {
                Log.i("MainActivity:Pytorch:", keys);
              }
              Tensor segOTensor = dict.get("out").toTensor();

              float[] output = segOTensor.getDataAsFloatArray();
              // Log.i("MainActivity:Pytorch:", String.format("%d", segOTensor.shape));
              result.success(dout);
              break;
            default:
              result.notImplemented();
              break;
          }
        }
      );
  }
}
