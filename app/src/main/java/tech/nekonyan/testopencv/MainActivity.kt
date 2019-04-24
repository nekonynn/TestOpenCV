package tech.nekonyan.testopencv

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.support.v4.app.ActivityCompat
import android.support.v4.content.ContextCompat
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import android.view.WindowManager
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.util.*

class MainActivity : Activity(), CvCameraViewListener2 {

    private var mItemPreviewRGBA: MenuItem? = null
    private var mItemPreviewHist: MenuItem? = null
    private var mItemPreviewCanny: MenuItem? = null
    private var mItemPreviewSepia: MenuItem? = null
    private var mItemPreviewSobel: MenuItem? = null
    private var mItemPreviewZoom: MenuItem? = null
    private var mItemPreviewPixelize: MenuItem? = null
    private var mItemPreviewPosterize: MenuItem? = null
    private var mOpenCvCameraView: CameraBridgeViewBase? = null

    private var mSize0: Size? = null

    private var mIntermediateMat: Mat? = null
    private var mMat0: Mat? = null
    private var mChannels: Array<MatOfInt>? = null
    private var mHistSize: MatOfInt? = null
    private val mHistSizeNum = 25
    private var mRanges: MatOfFloat? = null
    private var mColorsRGB: Array<Scalar>? = null
    private var mColorsHue: Array<Scalar>? = null
    private var mWhilte: Scalar? = null
    private var mP1: Point? = null
    private var mP2: Point? = null
    private var mBuff: FloatArray? = null
    private var mSepiaKernel: Mat? = null

    private val mLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                LoaderCallbackInterface.SUCCESS -> {
                    Log.i(TAG, "OpenCV loaded successfully")
                    mOpenCvCameraView!!.enableView()
                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    init {
        Log.i(TAG, "Instantiated new " + this.javaClass)
    }

    /** Called when the activity is first created.  */
    public override fun onCreate(savedInstanceState: Bundle?) {
        Log.i(TAG, "called onCreate")
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setContentView(R.layout.image_manipulations_surface_view)

        // First check android version
        //Check if permission is already granted
        //thisActivity is your activity. (e.g.: MainActivity.this)
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {

            // Give first an explanation, if needed.
            /*if (ActivityCompat.shouldShowRequestPermissionRationale(this,
                    Manifest.permission.CAMERA)) {

                // Show an explanation to the user *asynchronously* -- don't block
                // this thread waiting for the user's response! After the user
                // sees the explanation, try again to request the permission.

            } else {*/

            // No explanation needed, we can request the permission.
            ActivityCompat.requestPermissions(this,
                arrayOf(Manifest.permission.CALL_PHONE),
                100)
            // }
        }
        mOpenCvCameraView = findViewById(R.id.image_manipulations_activity_surface_view)
        mOpenCvCameraView!!.visibility = CameraBridgeViewBase.VISIBLE
        mOpenCvCameraView!!.setCvCameraViewListener(this)
    }

    public override fun onPause() {
        super.onPause()
        if (mOpenCvCameraView != null)
            mOpenCvCameraView!!.disableView()
    }

    public override fun onResume() {
        super.onResume()
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization")
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback)
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!")
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }
    }

    public override fun onDestroy() {
        super.onDestroy()
        if (mOpenCvCameraView != null)
            mOpenCvCameraView!!.disableView()
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        Log.i(TAG, "called onCreateOptionsMenu")
        mItemPreviewRGBA = menu.add("Preview RGBA")
        mItemPreviewHist = menu.add("Histograms")
        mItemPreviewCanny = menu.add("Canny")
        mItemPreviewSepia = menu.add("Sepia")
        mItemPreviewSobel = menu.add("Sobel")
        mItemPreviewZoom = menu.add("Zoom")
        mItemPreviewPixelize = menu.add("Pixelize")
        mItemPreviewPosterize = menu.add("Posterize")
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        Log.i(TAG, "called onOptionsItemSelected; selected item: $item")
        if (item === mItemPreviewRGBA)
            viewMode = VIEW_MODE_RGBA
        if (item === mItemPreviewHist)
            viewMode = VIEW_MODE_HIST
        else if (item === mItemPreviewCanny)
            viewMode = VIEW_MODE_CANNY
        else if (item === mItemPreviewSepia)
            viewMode = VIEW_MODE_SEPIA
        else if (item === mItemPreviewSobel)
            viewMode = VIEW_MODE_SOBEL
        else if (item === mItemPreviewZoom)
            viewMode = VIEW_MODE_ZOOM
        else if (item === mItemPreviewPixelize)
            viewMode = VIEW_MODE_PIXELIZE
        else if (item === mItemPreviewPosterize)
            viewMode = VIEW_MODE_POSTERIZE
        return true
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        mIntermediateMat = Mat()
        mSize0 = Size()
        mChannels = arrayOf(MatOfInt(0), MatOfInt(1), MatOfInt(2))
        mBuff = FloatArray(mHistSizeNum)
        mHistSize = MatOfInt(mHistSizeNum)
        mRanges = MatOfFloat(0f, 256f)
        mMat0 = Mat()
        mColorsRGB =
            arrayOf(Scalar(200.0, 0.0, 0.0, 255.0), Scalar(0.0, 200.0, 0.0, 255.0), Scalar(0.0, 0.0, 200.0, 255.0))
        mColorsHue = arrayOf(
            Scalar(255.0, 0.0, 0.0, 255.0),
            Scalar(255.0, 60.0, 0.0, 255.0),
            Scalar(255.0, 120.0, 0.0, 255.0),
            Scalar(255.0, 180.0, 0.0, 255.0),
            Scalar(255.0, 240.0, 0.0, 255.0),
            Scalar(215.0, 213.0, 0.0, 255.0),
            Scalar(150.0, 255.0, 0.0, 255.0),
            Scalar(85.0, 255.0, 0.0, 255.0),
            Scalar(20.0, 255.0, 0.0, 255.0),
            Scalar(0.0, 255.0, 30.0, 255.0),
            Scalar(0.0, 255.0, 85.0, 255.0),
            Scalar(0.0, 255.0, 150.0, 255.0),
            Scalar(0.0, 255.0, 215.0, 255.0),
            Scalar(0.0, 234.0, 255.0, 255.0),
            Scalar(0.0, 170.0, 255.0, 255.0),
            Scalar(0.0, 120.0, 255.0, 255.0),
            Scalar(0.0, 60.0, 255.0, 255.0),
            Scalar(0.0, 0.0, 255.0, 255.0),
            Scalar(64.0, 0.0, 255.0, 255.0),
            Scalar(120.0, 0.0, 255.0, 255.0),
            Scalar(180.0, 0.0, 255.0, 255.0),
            Scalar(255.0, 0.0, 255.0, 255.0),
            Scalar(255.0, 0.0, 215.0, 255.0),
            Scalar(255.0, 0.0, 85.0, 255.0),
            Scalar(255.0, 0.0, 0.0, 255.0)
        )
        mWhilte = Scalar.all(255.0)
        mP1 = Point()
        mP2 = Point()

        // Fill sepia kernel
        val varR = FloatArray(4) { i ->
            when (i) {
                0 -> 0.189F
                1 -> 0.769F
                2 -> 0.393F
                3 -> 0.0F
                else -> 0.0F
            }
        }

        val varG = FloatArray(4) { i ->
            when (i) {
                0 -> 0.168f
                1 -> 0.686f
                2 -> 0.349f
                3 -> 0.0F
                else -> 0.0F
            }
        }

        val varB = FloatArray(4) {i ->
            when (i) {
                0 -> 0.131f
                1 -> 0.534f
                2 -> 0.272f
                3 -> 0.0F
                else -> 0.0F
            }
        }

        val varA = FloatArray(4) {i ->
            when (i) {
                0 -> 0.0F
                1 -> 0.0F
                2 -> 0.0F
                3 -> 1F
                else -> 0.0F
            }
        }
        mSepiaKernel = Mat(4, 4, CvType.CV_32F)
        mSepiaKernel!!.put(0, 0, varR)
        mSepiaKernel!!.put(1, 0, varG)
        mSepiaKernel!!.put(2, 0, varB)
        mSepiaKernel!!.put(3, 0, varA)

        Log.d("TestOpenCV", Arrays.toString(varR))
        Log.d("TestOpenCV", Arrays.toString(varG))
        Log.d("TestOpenCV", Arrays.toString(varB))
        Log.d("TestOpenCV", Arrays.toString(varA))
    }

    override fun onCameraViewStopped() {
        // Explicitly deallocate Mats
        if (mIntermediateMat != null)
            mIntermediateMat!!.release()

        mIntermediateMat = null
    }

    override fun onCameraFrame(inputFrame: CvCameraViewFrame): Mat {
        val rgba = inputFrame.rgba()
        val sizeRgba = rgba.size()

        val rgbaInnerWindow: Mat

        val rows = sizeRgba.height.toInt()
        val cols = sizeRgba.width.toInt()

        val left = cols / 8
        val top = rows / 8

        val width = cols * 3 / 4
        val height = rows * 3 / 4

        when (viewMode) {
            VIEW_MODE_RGBA -> {
            }

            VIEW_MODE_HIST -> {
                val hist = Mat()
                var thikness = (sizeRgba.width / (mHistSizeNum + 10).toDouble() / 5.0).toInt()
                if (thikness > 5) thikness = 5
                val offset = ((sizeRgba.width - (5 * mHistSizeNum + 4 * 10) * thikness) / 2).toInt()
                // RGB
                for (c in 0..2) {
                    Imgproc.calcHist(Arrays.asList(rgba), mChannels!![c], mMat0!!, hist, mHistSize, mRanges)
                    Core.normalize(hist, hist, sizeRgba.height / 2, 0.0, Core.NORM_INF)
                    hist.get(0, 0, mBuff)
                    for (h in 0 until mHistSizeNum) {
                        mP2!!.x = (offset + (c * (mHistSizeNum + 10) + h) * thikness).toDouble()
                        mP1!!.x = mP2!!.x
                        mP1!!.y = sizeRgba.height - 1
                        mP2!!.y = mP1!!.y - 2.0 - mBuff!![h].toInt().toDouble()
                        Imgproc.line(rgba, mP1!!, mP2!!, mColorsRGB!![c], thikness)
                    }
                }
                // Value and Hue
                Imgproc.cvtColor(rgba, mIntermediateMat!!, Imgproc.COLOR_RGB2HSV_FULL)
                // Value
                Imgproc.calcHist(
                    Arrays.asList<Mat>(mIntermediateMat),
                    mChannels!![2],
                    mMat0!!,
                    hist,
                    mHistSize,
                    mRanges
                )
                Core.normalize(hist, hist, sizeRgba.height / 2, 0.0, Core.NORM_INF)
                hist.get(0, 0, mBuff)
                for (h in 0 until mHistSizeNum) {
                    mP2!!.x = (offset + (3 * (mHistSizeNum + 10) + h) * thikness).toDouble()
                    mP1!!.x = mP2!!.x
                    mP1!!.y = sizeRgba.height - 1
                    mP2!!.y = mP1!!.y - 2.0 - mBuff!![h].toInt().toDouble()
                    Imgproc.line(rgba, mP1!!, mP2!!, mWhilte!!, thikness)
                }
                // Hue
                Imgproc.calcHist(
                    Arrays.asList<Mat>(mIntermediateMat),
                    mChannels!![0],
                    mMat0!!,
                    hist,
                    mHistSize,
                    mRanges
                )
                Core.normalize(hist, hist, sizeRgba.height / 2, 0.0, Core.NORM_INF)
                hist.get(0, 0, mBuff)
                for (h in 0 until mHistSizeNum) {
                    mP2!!.x = (offset + (4 * (mHistSizeNum + 10) + h) * thikness).toDouble()
                    mP1!!.x = mP2!!.x
                    mP1!!.y = sizeRgba.height - 1
                    mP2!!.y = mP1!!.y - 2.0 - mBuff!![h].toInt().toDouble()
                    Imgproc.line(rgba, mP1!!, mP2!!, mColorsHue!![h], thikness)
                }
            }

            VIEW_MODE_CANNY -> {
                rgbaInnerWindow = rgba.submat(top, top + height, left, left + width)
                Imgproc.Canny(rgbaInnerWindow, mIntermediateMat!!, 80.0, 90.0)
                Imgproc.cvtColor(mIntermediateMat!!, rgbaInnerWindow, Imgproc.COLOR_GRAY2BGRA, 4)
                rgbaInnerWindow.release()
            }

            VIEW_MODE_SOBEL -> {
                val gray = inputFrame.gray()
                val grayInnerWindow = gray.submat(top, top + height, left, left + width)
                rgbaInnerWindow = rgba.submat(top, top + height, left, left + width)
                Imgproc.Sobel(grayInnerWindow, mIntermediateMat!!, CvType.CV_8U, 1, 1)
                Core.convertScaleAbs(mIntermediateMat!!, mIntermediateMat!!, 10.0, 0.0)
                Imgproc.cvtColor(mIntermediateMat!!, rgbaInnerWindow, Imgproc.COLOR_GRAY2BGRA, 4)
                grayInnerWindow.release()
                rgbaInnerWindow.release()
            }

            VIEW_MODE_SEPIA -> {
                rgbaInnerWindow = rgba.submat(top, top + height, left, left + width)
                Core.transform(rgbaInnerWindow, rgbaInnerWindow, mSepiaKernel!!)
                rgbaInnerWindow.release()
            }

            VIEW_MODE_ZOOM -> {
                val zoomCorner = rgba.submat(0, rows / 2 - rows / 10, 0, cols / 2 - cols / 10)
                val mZoomWindow = rgba.submat(
                    rows / 2 - 9 * rows / 100,
                    rows / 2 + 9 * rows / 100,
                    cols / 2 - 9 * cols / 100,
                    cols / 2 + 9 * cols / 100
                )
                Imgproc.resize(mZoomWindow, zoomCorner, zoomCorner.size(), 0.0, 0.0, Imgproc.INTER_LINEAR_EXACT)
                val wsize = mZoomWindow.size()
                Imgproc.rectangle(
                    mZoomWindow,
                    Point(1.0, 1.0),
                    Point(wsize.width - 2, wsize.height - 2),
                    Scalar(255.0, 0.0, 0.0, 255.0),
                    2
                )
                zoomCorner.release()
                mZoomWindow.release()
            }

            VIEW_MODE_PIXELIZE -> {
                rgbaInnerWindow = rgba.submat(top, top + height, left, left + width)
                Imgproc.resize(rgbaInnerWindow, mIntermediateMat!!, mSize0!!, 0.1, 0.1, Imgproc.INTER_NEAREST)
                Imgproc.resize(
                    mIntermediateMat!!,
                    rgbaInnerWindow,
                    rgbaInnerWindow.size(),
                    0.0,
                    0.0,
                    Imgproc.INTER_NEAREST
                )
                rgbaInnerWindow.release()
            }

            VIEW_MODE_POSTERIZE -> {
                /*
            Imgproc.cvtColor(rgbaInnerWindow, mIntermediateMat, Imgproc.COLOR_RGBA2RGB);
            Imgproc.pyrMeanShiftFiltering(mIntermediateMat, mIntermediateMat, 5, 50);
            Imgproc.cvtColor(mIntermediateMat, rgbaInnerWindow, Imgproc.COLOR_RGB2RGBA);
            */
                rgbaInnerWindow = rgba.submat(top, top + height, left, left + width)
                Imgproc.Canny(rgbaInnerWindow, mIntermediateMat!!, 80.0, 90.0)
                rgbaInnerWindow.setTo(Scalar(0.0, 0.0, 0.0, 255.0), mIntermediateMat)
                Core.convertScaleAbs(rgbaInnerWindow, mIntermediateMat!!, 1.0 / 16, 0.0)
                Core.convertScaleAbs(mIntermediateMat!!, rgbaInnerWindow, 16.0, 0.0)
                rgbaInnerWindow.release()
            }
        }

        return rgba
    }

    companion object {
        private const val TAG = "OCVSample::Activity"

        const val VIEW_MODE_RGBA = 0
        const val VIEW_MODE_HIST = 1
        const val VIEW_MODE_CANNY = 2
        const val VIEW_MODE_SEPIA = 3
        const val VIEW_MODE_SOBEL = 4
        const val VIEW_MODE_ZOOM = 5
        const val VIEW_MODE_PIXELIZE = 6
        const val VIEW_MODE_POSTERIZE = 7

        var viewMode = VIEW_MODE_SEPIA
    }
}

