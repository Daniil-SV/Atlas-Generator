#pragma once

#include <vector>
#include <map>

#include <opencv2/opencv.hpp>

#define LIBNEST2D_GEOMETRIES_clipper
#define LIBNEST2D_OPTIMIZER_nlopt
#include <libnest2d/libnest2d.hpp>

#define MinTextureDimension 512
#define MaxTextureDimension 4096

#define MinExtrude 1
#define MaxExtrude 5

#define MinScaleFactor 1
#define MaxScaleFactor 4

#define PercentOf(proc, num) (num * proc / 100)

using namespace std;
using libnest2d::ProgressFunction;

namespace sc {
	enum class AtlasGeneratorResult {
		OK = 0,
		BAD_POLYGON,
		TOO_MANY_IMAGES,
		TOO_BIG_IMAGE
	};
}

namespace sc {
	struct AtlasGeneratorConfig {
		enum class TextureType : int
		{
			RGBA = CV_8UC4,
			RGB = CV_8UC3,
			LUMINANCE_ALPHA = CV_8UC2,
			LIMINANCE = CV_8UC1
		};

		TextureType textureType = TextureType::RGBA;
		pair<uint16_t, uint16_t> maxSize = { 2048, 2048 };
		uint8_t scaleFactor = MinScaleFactor;
		uint8_t extrude = 2;

		libnest2d::NestControl control = {};
	};

	struct AtlasGeneratorVertex {
		AtlasGeneratorVertex() { };
		AtlasGeneratorVertex(uint16_t x, uint16_t y, uint16_t u, uint16_t v) {
			xy = { x, y };
			uv = { u, v };
		};

		pair<uint16_t, uint16_t> uv;
		pair<uint16_t, uint16_t> xy;
	};

	struct AtlasGeneratorItem {
		AtlasGeneratorItem(cv::Mat image) : image(image) {};

		cv::Mat image;
		uint8_t textureIndex = 0xFF;
		vector<AtlasGeneratorVertex> polygon;
	};

	struct AtlasGeneratorOutput {
		vector<cv::Mat> atlases;
		vector<AtlasGeneratorItem> items;
	};
}

namespace sc {
	class AtlasGenerator {
		// Some functions for drawing
#ifdef DEBUG
		inline static cv::RNG rng = cv::RNG(time(NULL));
		static void ShowImage(string name, cv::Mat image) {
			cv::namedWindow(name, cv::WINDOW_NORMAL);

			cv::imshow(name, image);
			cv::waitKey(0);
		}

		static void ShowContour(cv::Mat src, vector<sc::AtlasGeneratorVertex> points) {
			vector<cv::Point> cvPoints;
			for (auto& point : points) {
				cvPoints.push_back({ point.uv.first, point.uv.second });
			}

			ShowContour(src, cvPoints);
		}

		static void ShowContour(cv::Mat src, vector<cv::Point> points) {
			cv::Mat drawing = src.clone();
			drawContours(drawing, vector<vector<cv::Point>>(1, points), 0, cv::Scalar(255, 255, 255), (int)PercentOf(2, (drawing.rows + drawing.cols) / 2), cv::LINE_AA);

			for (cv::Point& point : points) {
				circle(drawing, point, (int)PercentOf(5, (drawing.rows + drawing.cols) / 5), { 0, 0, 255 }, (int)PercentOf(3, (drawing.rows + drawing.cols) / 2), cv::LINE_AA);
			}
			ShowImage("Image polygon", drawing);
			cv::destroyAllWindows();
		}
#endif
		static void NormalizeConfig(AtlasGeneratorConfig& config);

		static vector<cv::Point> GetImageContour(cv::Mat& image);

		static cv::Mat GetImagePolygon(AtlasGeneratorItem& item, AtlasGeneratorConfig& config);

		static cv::Mat ImagePreprocess(cv::Mat& src);
		static cv::Mat MaskPreprocess(cv::Mat& src);

		static void SnapToBorder(cv::Mat src, vector<cv::Point>& points);
		static void ExtrudePoints(cv::Mat src, vector<cv::Point>& points);

		static bool IsRectangle(cv::Mat& image, AtlasGeneratorConfig& config);

		static void PlaceImageTo(cv::Mat& src, cv::Mat& dst, uint16_t x, uint16_t y);

		static uint32_t SearchDuplicate(vector<AtlasGeneratorItem>& items, cv::Mat& image, uint32_t range);

		static bool CompareImage(cv::Mat src1, cv::Mat src2);

	public:
		static AtlasGeneratorResult Generate(vector<AtlasGeneratorItem>& items, vector<cv::Mat>& atlases, AtlasGeneratorConfig& config);
	};
}