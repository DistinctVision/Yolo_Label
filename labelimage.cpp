#include "labelimage.h"
#include <QPainter>
#include <math.h>       /* fabs */

using std::ifstream;

QColor LabelImage::BOX_COLORS[10] ={  Qt::green,
        Qt::darkGreen,
        Qt::blue,
        Qt::darkBlue,
        Qt::yellow,
        Qt::darkYellow,
        Qt::red,
        Qt::darkRed,
        Qt::cyan,
        Qt::darkCyan};

LabelImage::LabelImage(QWidget *parent)
    :QLabel(parent)
{
    init();
}

void LabelImage::mouseMoveEvent(QMouseEvent *ev)
{
    setMousePosition(ev->x(), ev->y());

    showImage();
    emit Mouse_Moved();
}

void LabelImage::mousePressEvent(QMouseEvent *ev)
{
    setMousePosition(ev->x(), ev->y());

    if(ev->button() == Qt::RightButton)
    {
        removeFocusedObjectBox(m_relative_mouse_pos_in_ui);
        showImage();
    }
    else if(ev->button() == Qt::LeftButton)
    {
        if(m_bLabelingStarted == false)
        {
            m_relatvie_mouse_pos_LBtnClicked_in_ui      = m_relative_mouse_pos_in_ui;
            m_bLabelingStarted                          = true;
        }
        else
        {
            ObjectLabelingBox objBoundingbox;

            objBoundingbox.label    = m_focusedObjectLabel;
            objBoundingbox.box      = getRelativeRectFromTwoPoints(m_relative_mouse_pos_in_ui,
                                                                   m_relatvie_mouse_pos_LBtnClicked_in_ui);

            bool width_is_too_small     = objBoundingbox.box.width() * m_inputImg.width()  < 4;
            bool height_is_too_small    = objBoundingbox.box.height() * m_inputImg.height() < 4;

            if(!width_is_too_small && !height_is_too_small)
                m_objBoundingBoxes.push_back(objBoundingbox);

            m_bLabelingStarted              = false;

            showImage();
        }
    }

    emit Mouse_Pressed();
}

void LabelImage::mouseReleaseEvent(QMouseEvent *ev)
{
    emit Mouse_Release();
}

void LabelImage::init()
{
    m_objBoundingBoxes.clear();
    m_bLabelingStarted              = false;
    m_focusedObjectLabel            = 0;

    QPoint mousePosInUi = this->mapFromGlobal(QCursor::pos());
    bool mouse_is_in_image = QRect(0, 0, this->width(), this->height()).contains(mousePosInUi);

    if  (mouse_is_in_image)
    {
        setMousePosition(mousePosInUi.x(), mousePosInUi.y());
    }
    else
    {
        setMousePosition(0., 0.);
    }
}

void LabelImage::setMousePosition(int x, int y)
{
    if(x < 0) x = 0;
    if(y < 0) y = 0;

    if(x > this->width())   x = this->width() - 1;
    if(y > this->height())  y = this->height() - 1;

    m_relative_mouse_pos_in_ui = cvtAbsoluteToRelativePoint(QPoint(x, y));
}

bool LabelImage::openImage(const QString & qstrImg)
{
    m_detectedObjects.clear();
    QImage img(qstrImg);

    if (img.isNull())
    {
        m_inputImg = QImage();
        return  false;
    }
    m_objBoundingBoxes.clear();

    m_inputImg          = img.copy();
    m_inputImg          = m_inputImg.convertToFormat(QImage::Format_RGB888);

    m_bLabelingStarted  = false;

    QPoint mousePosInUi     = this->mapFromGlobal(QCursor::pos());
    bool mouse_is_in_image  = QRect(0, 0, this->width(), this->height()).contains(mousePosInUi);

    if  (mouse_is_in_image)
    {
        setMousePosition(mousePosInUi.x(), mousePosInUi.y());
    }
    else
    {
        setMousePosition(0., 0.);
    }

    return true;
}

bool LabelImage::setCvImage(const cv::Mat & cvImage)
{
    m_detectedObjects.clear();
    if (cvImage.empty())
    {
        m_inputImg = QImage();
        return  false;
    }
    QImage img(cvImage.data, cvImage.size().width, cvImage.size().height, QImage::Format_RGB888);
    m_objBoundingBoxes.clear();

    m_inputImg          = img.copy().convertToFormat(QImage::Format_RGB888);

    m_bLabelingStarted  = false;

    QPoint mousePosInUi     = this->mapFromGlobal(QCursor::pos());
    bool mouse_is_in_image  = QRect(0, 0, this->width(), this->height()).contains(mousePosInUi);

    if  (mouse_is_in_image)
    {
        setMousePosition(mousePosInUi.x(), mousePosInUi.y());
    }
    else
    {
        setMousePosition(0., 0.);
    }

    return true;
}

void LabelImage::showImage()
{
    if (m_inputImg.isNull())
        return;

    QImage imageOnUi = m_inputImg.scaled(this->width(), this->height(),Qt::IgnoreAspectRatio,Qt::SmoothTransformation);

    QPainter painter(&imageOnUi);

    int penThick = 3;

    QColor crossLineColor(255, 187, 0);

    drawDetectedObjects(painter, QColor(255, 0, 0, 100));
    drawCrossLine(painter, crossLineColor, penThick);
    drawFocusedObjectBox(painter, Qt::magenta, penThick);
    drawObjectBoxes(painter, penThick);

    this->setPixmap(QPixmap::fromImage(imageOnUi));
}

QVector<ObjectLabelingBox> LabelImage::objBoundingBoxes() const
{
    return m_objBoundingBoxes;
}

void LabelImage::resetObjBoundingBoxes()
{
    m_objBoundingBoxes.clear();
}

void LabelImage::loadLabelData(const QString& labelFilePath)
{
    ifstream inputFile(labelFilePath.toStdString());

    if(inputFile.is_open())
    {
        double          inputFileValue;
        QVector<double> inputFileValues;

        while(inputFile >> inputFileValue)
            inputFileValues.push_back(inputFileValue);

        for(int i = 0; i < inputFileValues.size(); i += 5)
        {
            try {
                ObjectLabelingBox objBox;

                objBox.label = static_cast<int>(inputFileValues.at(i));

                double midX     = inputFileValues.at(i + 1);
                double midY     = inputFileValues.at(i + 2);
                double width    = inputFileValues.at(i + 3);
                double height   = inputFileValues.at(i + 4);

                double leftX    = midX - width/2.;
                double topY     = midY - height/2.;

                objBox.box.setX(leftX); // translation: midX -> leftX
                objBox.box.setY(topY); // translation: midY -> topY
                objBox.box.setWidth(width);
                objBox.box.setHeight(height);

                m_objBoundingBoxes.push_back(objBox);
            }
            catch (const std::out_of_range& e) {
                std::cout << "loadLabelData: Out of Range error.";
            }
        }
    }
}

void LabelImage::setDrawObjectBoxColors(const QVector<QColor> &drawObjectBoxColor)
{
    m_drawObjectBoxColor = drawObjectBoxColor;
}

void LabelImage::setDetectedObjects(const QVector<ObjectLabelingBox> &detectedObjects)
{
    m_detectedObjects = detectedObjects;
}

void LabelImage::setFocusObjectLabel(int nLabel)
{
    m_focusedObjectLabel = nLabel;
}

void LabelImage::setFocusObjectName(QString qstrName)
{
    m_foucsedObjectName = qstrName;
}

bool LabelImage::isOpened()
{
    return !m_inputImg.isNull();
}

QImage LabelImage::crop(QRect rect)
{
    return m_inputImg.copy(rect);
}

void LabelImage::drawCrossLine(QPainter& painter, QColor color, int thickWidth)
{
    if(m_relative_mouse_pos_in_ui == QPointF(0., 0.)) return;

    QPen pen;
    pen.setWidth(thickWidth);

    pen.setColor(color);
    painter.setPen(pen);

    QPoint absolutePoint = cvtRelativeToAbsolutePoint(m_relative_mouse_pos_in_ui);

    std::cout <<"absolutePoint.x() = "<< absolutePoint.x() << std::endl;
    //draw cross line
    painter.drawLine(QPoint(absolutePoint.x(),0), QPoint(absolutePoint.x(), this->height() - 1));
    painter.drawLine(QPoint(0,absolutePoint.y()), QPoint(this->width() - 1, absolutePoint.y()));

    std::cout << "draw Cross Line" << std::endl;
}

void LabelImage::drawFocusedObjectBox(QPainter& painter, Qt::GlobalColor color, int thickWidth)
{
    if(m_bLabelingStarted == true)
    {
        QPen pen;
        pen.setWidth(thickWidth);

        pen.setColor(color);
        painter.setPen(pen);

        //relative coord to absolute coord

        QPoint absolutePoint1 = cvtRelativeToAbsolutePoint(m_relatvie_mouse_pos_LBtnClicked_in_ui);
        QPoint absolutePoint2 = cvtRelativeToAbsolutePoint(m_relative_mouse_pos_in_ui);

        painter.drawRect(QRect(absolutePoint1, absolutePoint2));
    }
}

void LabelImage::drawObjectBoxes(QPainter& painter, int thickWidth)
{
    QPen pen;
    pen.setWidth(thickWidth);

    for(ObjectLabelingBox boundingbox: m_objBoundingBoxes)
    {
        pen.setColor(m_drawObjectBoxColor.at(boundingbox.label));
        painter.setPen(pen);

        painter.drawRect(cvtRelativeToAbsoluteRectInUi(boundingbox.box));
    }
}

void LabelImage::drawDetectedObjects(QPainter& painter, const QColor & color)
{
    QPen pen;
    pen.setWidth(2);
    pen.setColor(color);
    painter.setPen(pen);
    for (const ObjectLabelingBox & obj: m_detectedObjects)
    {
        painter.drawRect(cvtRelativeToAbsoluteRectInUi(obj.box));
    }
}

void LabelImage::removeFocusedObjectBox(QPointF point)
{
    int     removeBoxIdx = -1;
    double  nearestBoxDistance   = 99999999999999.;

    for(int i = 0; i < m_objBoundingBoxes.size(); i++)
    {
        QRectF objBox = m_objBoundingBoxes.at(i).box;

        if(objBox.contains(point))
        {
            double distance = objBox.width() + objBox.height();
            if(distance < nearestBoxDistance)
            {
                nearestBoxDistance = distance;
                removeBoxIdx = i;
            }
        }
    }

    if(removeBoxIdx != -1)
    {
        m_objBoundingBoxes.remove(removeBoxIdx);
    }
}

QRectF LabelImage::getRelativeRectFromTwoPoints(QPointF p1, QPointF p2)
{
    double midX    = (p1.x() + p2.x()) / 2.;
    double midY    = (p1.y() + p2.y()) / 2.;
    double width   = fabs(p1.x() - p2.x());
    double height  = fabs(p1.y() - p2.y());

    QPointF topLeftPoint(midX - width/2., midY - height/2.);
    QPointF bottomRightPoint(midX + width/2., midY + height/2.);

    return QRectF(topLeftPoint, bottomRightPoint);
}

QRect LabelImage::cvtRelativeToAbsoluteRectInUi(QRectF rectF)
{
    return QRect(static_cast<int>(rectF.x() * this->width() + 0.5),
                 static_cast<int>(rectF.y() * this->height()+ 0.5),
                 static_cast<int>(rectF.width() * this->width()+ 0.5),
                 static_cast<int>(rectF.height()* this->height()+ 0.5));
}

QRect LabelImage::cvtRelativeToAbsoluteRectInImage(QRectF rectF)
{
    return QRect(static_cast<int>(rectF.x() * m_inputImg.width()),
                 static_cast<int>(rectF.y() * m_inputImg.height()),
                 static_cast<int>(rectF.width() * m_inputImg.width()),
                 static_cast<int>(rectF.height()* m_inputImg.height()));
}

QPoint LabelImage::cvtRelativeToAbsolutePoint(QPointF p)
{
    return QPoint(static_cast<int>(p.x() * this->width() + 0.5), static_cast<int>(p.y() * this->height() + 0.5));
}

QPointF LabelImage::cvtAbsoluteToRelativePoint(QPoint p)
{
    return QPointF(static_cast<double>(p.x()) / this->width(), static_cast<double>(p.y()) / this->height());
}

QVector<QColor> LabelImage::getDrawObjectBoxColors() const
{
    return m_drawObjectBoxColor;
}
