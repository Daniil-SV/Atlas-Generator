#include "AtlasGenerator.h"
#include "AtlasGeneratorPlacer.hpp"

#define PercentOf(proc, num) (num * proc / 100)

namespace sc {
	void AtlasGenerator::NormalizeConfig(AtlasGeneratorConfig& config)
	{
		if (config.maxSize.first > MaxTextureDimension) {
			config.maxSize.first = MaxTextureDimension;
		}
		else if (config.maxSize.first < MinTextureDimension) {
			config.maxSize.first = MinTextureDimension;
		}

		if (config.maxSize.second > MaxTextureDimension) {
			config.maxSize.second = MaxTextureDimension;
		}
		else if (config.maxSize.second < MinTextureDimension) {
			config.maxSize.second = MinTextureDimension;
		}

		if (config.extrude > MaxExtrude) {
			config.extrude = MaxExtrude;
		}
		else if (config.extrude < MinExtrude) {
			config.extrude = MinExtrude;
		}
	};

	vector<cv::Point> AtlasGenerator::GetImageContour(cv::Mat& image)
	{
		using namespace cv;

		vector<Point> result;
		vector<vector<Point>> contours;

		findContours(image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		const double approxMultiplier = 0.0015;
		if (contours.size() == 1) {
			double perim = arcLength(contours[0], true);
			approxPolyDP(contours[0], result, approxMultiplier * perim, true);
			result = contours[0];
		}
		else {
			for (vector<Point>& points : contours) {
				vector<Point> reduced;
				double perim = arcLength(points, true);
				approxPolyDP(points, reduced, approxMultiplier * perim, true);

				for (Point& point : reduced) {
					result.push_back(point);
				}
			}
		}

		return result;
	}

	void AtlasGenerator::SnapPoints(cv::Mat src, vector<cv::Point>& points) {
		using namespace cv;

		const uint8_t offsetPercent = 0;
		const uint8_t snapPercent = 7;

		uint16_t minW = src.cols * snapPercent / 100;
		uint16_t maxW = src.cols - minW;

		uint16_t minH = src.rows * snapPercent / 100;
		uint16_t maxH = src.rows - minH;

		for (Point& point : points) {
			if (point.x < minW || point.x > maxW) {
				uint16_t offset = PercentOf(offsetPercent, src.rows);

				if (point.y > src.rows / 2) {
					if (point.y + offset > src.rows) {
						point.y = src.rows;
					}
					else {
						point.y += offset;
					}
				}
				else {
					if (0 > point.y - offset) {
						point.y = 0;
					}
					else {
						point.y -= offset;
					}
				}

				if (point.x < minW) {
					point.x = 0;
				}
				else {
					point.x = src.cols;
				}
			}

			if (point.y < minH || point.y > maxH) {
				uint16_t offset = PercentOf(offsetPercent, src.cols);

				if (point.x > src.cols / 2) {
					if (point.x + offset > src.cols) {
						point.x = src.cols;
					}
					else {
						point.x += offset;
					}
				}
				else {
					if (0 > point.x - offset) {
						point.x = 0;
					}
					else {
						point.x -= offset;
					}
				}

				if (point.y < minH) {
					point.y = 0;
				}
				else {
					point.y = src.rows;
				}
			}
		}
	}

	AtlasGeneratorResult AtlasGenerator::GetImagePolygon(AtlasGeneratorItem& item, AtlasGeneratorConfig& config)
	{
		using namespace cv;

		Mat imageMask;
		switch (item.image.channels())
		{
		case 4:
			extractChannel(item.image, imageMask, 3);
			break;
		case 2:
			extractChannel(item.image, imageMask, 1);
			break;
		default:
			imageMask = Mat(item.image.size(), CV_8UC1, Scalar(255));
			break;
		}

		Rect imageBounds = boundingRect(imageMask);

		Size srcSize = item.image.size();
		Size dstSize = imageMask.size();

		item.image = item.image(imageBounds);
		imageMask = imageMask(imageBounds);

		if (IsRectangle(item.image, config)) {
			item.polygon = vector<AtlasGeneratorVertex>(4);

			item.polygon[0].uv = { 0, 0 };
			item.polygon[1].uv = { 0, item.image.rows };
			item.polygon[2].uv = { item.image.cols, item.image.rows };
			item.polygon[3].uv = { item.image.cols, 0 };

			item.polygon[0].xy = { 0, 0 }; 
			item.polygon[1].xy = { 0, item.image.rows };
			item.polygon[2].xy = { item.image.cols, item.image.rows }; 
			item.polygon[3].xy = { item.image.cols, 0 }; 

			return AtlasGeneratorResult::OK;
		}
		else {
			item.polygon.clear();
		}

		vector<Point> contour = GetImageContour(imageMask);
#ifdef CV_DEBUG
		ShowContour(item.image, contour);
#endif
		vector<Point> contourHull;
		vector<Point> polygon;

		SnapPoints(imageMask, contour);

		convexHull(contour, polygon, true);
		

#ifdef CV_DEBUG
		ShowContour(item.image, polygon);
#endif

		for (const Point point : polygon) {
			item.polygon.push_back(
				AtlasGeneratorVertex
				(point.x + imageBounds.x, point.y + imageBounds.y, point.x, point.y)
			);
		}

#ifdef CV_DEBUG
		ShowContour(item.image, item.polygon);
#endif

		return AtlasGeneratorResult::OK;
	};

	bool AtlasGenerator::IsRectangle(cv::Mat& image, AtlasGeneratorConfig& config)
	{
		using namespace cv;

		Size size = image.size();
		int channels = image.channels();

		if (size.width < (config.maxSize.first * 3 / 100) && size.height < (config.maxSize.second * 3 / 100)) {
			return true;
		}

		// Image has no alpha. It makes no sense to generate a polygon
		if (channels == 1 || channels == 3) {
			return true;
		}

		return false;
	};

	void AtlasGenerator::PlaceImage(cv::Mat& src, cv::Mat& dst, uint16_t x, uint16_t y)
	{
		using namespace cv;

		Size srcSize = src.size();
		Size dstSize = dst.size();

		for (uint16_t h = 0; srcSize.height > h; h++) {
			uint16_t dstH = h + y;
			if (dstH >= dstSize.height) continue;

			for (uint16_t w = 0; srcSize.width > w; w++) {
				uint16_t dstW = w + x;
				if (dstW >= dstSize.width) continue;

				Vec4b pixel(0, 0, 0, 0);

				switch (src.channels())
				{
				case 4:
					pixel = src.at<cv::Vec4b>(h, w);
					break;
				default:
					break;
				}

				if (pixel[3] == 0) {
					continue;
				}

				switch (dst.channels())
				{
				case 4:
					dst.at<Vec4b>(dstH, dstW) = pixel;
					break;
				default:
					break;
				}
			}
		}
	};

	AtlasGeneratorResult AtlasGenerator::Generate(vector<AtlasGeneratorItem>& items, vector<cv::Mat>& atlases, AtlasGeneratorConfig& config) {
		using namespace libnest2d;
		NormalizeConfig(config);

		// Duplicated images
		vector<pair<uint32_t, uint32_t>> duplicates;
		//	 Src image index, dst image offset

		// Vector with polygons for libnest2d
		vector<Item> packerItems;
		for (uint32_t i = 0; items.size() > i; i++) {
			AtlasGeneratorItem& item = items[i];

			uint32_t imageIndex = GetImageIndex(items, item.image, i);

			if (imageIndex == UINT32_MAX) {
				// Polygon generation
				GetImagePolygon(item, config);

				if (item.polygon.size() <= 0) {
					return AtlasGeneratorResult::BAD_POLYGON;
				}

				// Adding new items to packer
				libnest2d::Item packerItem = libnest2d::Item(vector<ClipperLib::IntPoint>(item.polygon.size() + 1), {});

				for (uint16_t p = 0; packerItem.vertexCount() > p; p++) {
					if (p == item.polygon.size()) { // End point for libnest
						packerItem.setVertex(p, { item.polygon[0].uv.first, item.polygon[0].uv.second });
					}
					else {
						packerItem.setVertex(p, { item.polygon[p].uv.first, item.polygon[p].uv.second });
					}
				}

				packerItems.push_back(packerItem);
			}
			else {
				duplicates.push_back({ imageIndex, i });
			}
		}

		NestConfig<BottomLeftPlacer, DJDHeuristic> cfg;
		cfg.placer_config.epsilon = config.extrude;
		cfg.placer_config.allow_rotations = true;

		size_t binCount = nest(packerItems, Box(config.maxSize.first, config.maxSize.second, { config.maxSize.first / 2, config.maxSize.second / 2 }), config.extrude, cfg, config.control);
		if (binCount >= 0xFF) return AtlasGeneratorResult::TOO_MANY_IMAGES;

		// Texture preparing
		vector<cv::Size> textureSize(binCount);
		for (size_t i = 0; packerItems.size() > i; i++) {
			Item& item = packerItems[i];
			if (item.binId() == libnest2d::BIN_ID_UNSET) return AtlasGeneratorResult::BAD_POLYGON;

			auto shape = item.transformedShape();
			auto box = item.boundingBox();
			
			cv::Size& size = textureSize[item.binId()];

			auto x = getX(box.maxCorner());
			auto y = getY(box.maxCorner());

			if (x > size.height) {
				size.height = x;
			}
			if (y > size.width) {
				size.width = y;
			}
		}

		for (cv::Size& size : textureSize)
		{
			atlases.push_back(
				cv::Mat(
					size.width,
					size.height,
					(int)config.textureType,
					cv::Scalar(0)
				)
			);
		}

		for (uint32_t i = 0; items.size() > i; i++) {
			Item& packerItem = packerItems[i];
			AtlasGeneratorItem& item = items[i];
			cv::Mat& atlas = atlases[packerItem.binId()];

			auto rotation = packerItem.rotation();
			double rotationAngle = -(rotation.toDegrees());

			auto shape = packerItem.transformedShape();
			auto box = packerItem.boundingBox();

			// Point processing
			item.textureIndex = static_cast<uint8_t>(packerItem.binId());
			for (size_t j = 0; item.polygon.size() > j; j++) {
				uint16_t x = static_cast<uint16_t>(shape.Contour[j + 1].X);
				uint16_t y = static_cast<uint16_t>(shape.Contour[j + 1].Y);
				item.polygon[j].uv = {x, y};
			}

			// Image processing
			cv::Mat atlasSprite = item.image.clone();
			if (rotationAngle != 0) {
				cv::Point2f center((float)((atlasSprite.cols - 1) / 2.0), (float)((atlasSprite.rows - 1) / 2.0));
				cv::Mat rot = cv::getRotationMatrix2D(center, rotationAngle, 1.0);
				cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), atlasSprite.size(), (float)rotationAngle).boundingRect2f();

				rot.at<double>(0, 2) += bbox.width / 2.0 - item.image.cols / 2.0;
				rot.at<double>(1, 2) += bbox.height / 2.0 - item.image.rows / 2.0;

				cv::warpAffine(atlasSprite, atlasSprite, rot, bbox.size(), cv::INTER_NEAREST);
			}

			cv::copyMakeBorder(atlasSprite, atlasSprite, config.extrude, config.extrude, config.extrude, config.extrude, cv::BORDER_REPLICATE);

			auto x = getX(box.minCorner());
			auto y = getY(box.minCorner());
			PlaceImage(
				atlasSprite,
				atlas,
				static_cast<uint16_t>(x - config.extrude),
				static_cast<uint16_t>(y - config.extrude)
			);
		}

#ifdef CV_DEBUG
		vector<cv::Mat> sheets;
		cv::RNG rng = cv::RNG(time(NULL));

		for (cv::Mat& atlas : atlases) {
			sheets.push_back(cv::Mat(atlas.size(), CV_8UC4, cv::Scalar(0)));
		}

		for (AtlasGeneratorItem& item : items) {
			vector<cv::Point> polyContour;
			for (AtlasGeneratorVertex point : item.polygon) {
				polyContour.push_back(cv::Point(point.uv.first, point.uv.second));
			}
			cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			fillPoly(sheets[item.textureIndex], polyContour, color);
		}

		for (cv::Mat& sheet : sheets) {
			ShowImage("Sheet", sheet);
		}

		for (cv::Mat& atlas : atlases) {
			ShowImage("Atlas", atlas);
		}

		cv::destroyAllWindows();
#endif // DEBUG

		return AtlasGeneratorResult::OK;
	}
}