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

	AtlasGeneratorResult AtlasGenerator::GetImagePolygon(AtlasGeneratorItem& item, cv::Mat& image, AtlasGeneratorConfig& config)
	{
		using namespace cv;

		Mat polygonMask;
		switch (item.image.channels())
		{
		case 4:
			extractChannel(item.image, polygonMask, 3);
			break;
		case 2:
			extractChannel(item.image, polygonMask, 1);
			break;
		default:
			polygonMask = Mat(item.image.size(), CV_8UC1, Scalar(255));
			break;
		}

		Rect imageBounds = boundingRect(polygonMask);

		Size srcSize = item.image.size();
		Size dstSize = polygonMask.size();

		image = item.image(imageBounds);
		polygonMask = polygonMask(imageBounds);

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

		vector<Point> contour = GetImageContour(polygonMask);
#ifdef CV_DEBUG
		ShowContour(item.image, contour);
#endif
		vector<Point> contourHull;
		vector<Point> polygon;

		SnapPoints(polygonMask, contour);

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

	uint32_t AtlasGenerator::GetImageIndex(vector<AtlasGeneratorItem>& items, cv::Mat& image, uint32_t range) {
		using namespace cv;

		for (uint32_t i = 0; range > i; i++) {
			Mat& other = items[i].image;

			if (CompareImage(image, other)) {
				return i;
			}
		}

		return UINT32_MAX;
	}

	bool AtlasGenerator::CompareImage(cv::Mat src1, cv::Mat src2) {
		using namespace cv;

		if (src1.cols != src2.cols || src1.rows != src2.rows) return false;
		int imageChannelsCount = src1.channels();
		int otherChannelsCount = src2.channels();

		if (imageChannelsCount != otherChannelsCount) return false;

		vector<Mat> channels(imageChannelsCount);
		vector<Mat> otherChannels(imageChannelsCount);
		split(src1, channels);
		split(src2, otherChannels);

		size_t pixelCount = src1.rows * src1.cols;
		for (int j = 0; imageChannelsCount > j; j++) {
			for (int w = 0; src1.cols > w; w++) {
				for (int h = 0; src1.rows > h; h++) {
					uchar pix = channels[j].at<uchar>(h, w);
					uchar otherPix = otherChannels[j].at<uchar>(h, w);
					if (pix != otherPix) {
						return false;
					}
				}
			}

		}

		return true;
	}

	AtlasGeneratorResult AtlasGenerator::Generate(vector<AtlasGeneratorItem>& items, vector<cv::Mat>& atlases, AtlasGeneratorConfig& config) {
		using namespace libnest2d;
		NormalizeConfig(config);

		// Duplicated images
		vector<size_t> duplicates;

		// Vector with polygons for libnest2d
		vector<Item> packerItems;

		// Croped images
		vector<cv::Mat> images;

		for (uint32_t i = 0; items.size() > i; i++) {
			AtlasGeneratorItem& item = items[i];
			uint32_t imageIndex = GetImageIndex(items, item.image, i);

			if (imageIndex == UINT32_MAX) {
				// Polygon generation
				cv::Mat polygonImage;
				GetImagePolygon(item, polygonImage, config);
				images.push_back(polygonImage);

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
				duplicates.push_back(SIZE_MAX);
			}
			else {
				duplicates.push_back(imageIndex);
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
				size.height = (int)x;
			}
			if (y > size.width) {
				size.width = (int)y;
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

		uint32_t itemOffset = 0;
		for (uint32_t i = 0; items.size() > i; i++) {
			bool isDuplicate = duplicates[i] != SIZE_MAX;
			if (isDuplicate) {
				items[i] = items[duplicates[i]];
				itemOffset++;
				continue;
			}

			Item& packerItem = packerItems[i - itemOffset];
			AtlasGeneratorItem& item = items[i];
			cv::Mat& atlas = atlases[packerItem.binId()];

			auto rotation = packerItem.rotation();
			double rotationAngle = -(rotation.toDegrees());

			auto shape = packerItem.transformedShape();
			auto box = packerItem.boundingBox();

			// Point processing
			item.textureIndex = static_cast<uint8_t>(packerItem.binId());
			for (size_t j = 0; item.polygon.size() > j; j++) {
				
				uint16_t x = item.polygon[j].uv.first;
				uint16_t y = item.polygon[j].uv.second;

				uint16_t u = (uint16_t)ceil(x * rotation.cos() - y * rotation.sin() + getX(packerItem.translation()) );
				uint16_t v = (uint16_t)ceil(y * rotation.cos() + x * rotation.sin() + getY(packerItem.translation()) );

				item.polygon[j].uv = {u, v};
			}

			// Image processing
			cv::Mat sprite = images[i - itemOffset].clone();
			if (rotationAngle != 0) {
				cv::Point2f center((float)((sprite.cols - 1) / 2.0), (float)((sprite.rows - 1) / 2.0));
				cv::Mat rot = cv::getRotationMatrix2D(center, rotationAngle, 1.0);
				cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), sprite.size(), (float)rotationAngle).boundingRect2f();

				rot.at<double>(0, 2) += bbox.width / 2.0 - sprite.cols / 2.0;
				rot.at<double>(1, 2) += bbox.height / 2.0 - sprite.rows / 2.0;

				cv::warpAffine(sprite, sprite, rot, bbox.size(), cv::INTER_NEAREST);
			}

			cv::copyMakeBorder(sprite, sprite, config.extrude, config.extrude, config.extrude, config.extrude, cv::BORDER_REPLICATE);

			auto x = getX(box.minCorner());
			auto y = getY(box.minCorner());
			PlaceImage(
				sprite,
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