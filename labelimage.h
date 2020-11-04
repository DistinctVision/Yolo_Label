#ifndef LABEL_IMG_H
#define LABEL_IMG_H

#include <iostream>
#include <fstream>

#include <QObject>
#include <QVector>
#include <QLabel>
#include <QImage>
#include <QMouseEvent>
#include <QRectF>

#include <opencv2/core.hpp>

struct ObjectLabelingBox
{
    int     label;
    QRectF  box;
};

class LabelImage : public QLabel
{
    Q_OBJECT
public:
    LabelImage(QWidget *parent = nullptr);

    void mouseMoveEvent(QMouseEvent *ev);
    void mousePressEvent(QMouseEvent *ev);
    void mouseReleaseEvent(QMouseEvent *ev);

    static QColor BOX_COLORS[10];

    void init();
    bool openImage(const QString &);
    bool setCvImage(const cv::Mat & cvImage);
    void showImage();

    QVector<ObjectLabelingBox> objBoundingBoxes() const;
    void resetObjBoundingBoxes();

    void loadLabelData(const QString & );

    void setDetectedObjects(const QVector<ObjectLabelingBox> &detectedObjects);

    void setFocusObjectLabel(int);
    void setFocusObjectName(QString);

    bool isOpened();
    QImage crop(QRect);

    QRectF  getRelativeRectFromTwoPoints(QPointF , QPointF);

    QRect   cvtRelativeToAbsoluteRectInUi(QRectF);
    QRect   cvtRelativeToAbsoluteRectInImage(QRectF);

    QPoint  cvtRelativeToAbsolutePoint(QPointF);
    QPointF cvtAbsoluteToRelativePoint(QPoint);

    QVector<QColor> getDrawObjectBoxColors() const;
    void setDrawObjectBoxColors(const QVector<QColor> &drawObjectBoxColor);

signals:
    void Mouse_Moved();
    void Mouse_Pressed();
    void Mouse_Release();

private:
    QVector<ObjectLabelingBox> m_objBoundingBoxes;

    QVector<QColor> m_drawObjectBoxColor;

    int m_uiX;
    int m_uiY;

    int m_imgX;
    int m_imgY;

    bool m_bLabelingStarted;

    int             m_focusedObjectLabel;
    QString         m_foucsedObjectName;

    double m_aspectRatioWidth;
    double m_aspectRatioHeight;

    QImage m_inputImg;
    QVector<ObjectLabelingBox> m_detectedObjects;

    QPointF m_relative_mouse_pos_in_ui;
    QPointF m_relatvie_mouse_pos_LBtnClicked_in_ui;

    void setMousePosition(int , int);

    void drawCrossLine(QPainter& , QColor , int thickWidth = 3);
    void drawFocusedObjectBox(QPainter& , Qt::GlobalColor , int thickWidth = 3);
    void drawObjectBoxes(QPainter& , int thickWidth = 3);
    void drawDetectedObjects(QPainter& , const QColor &);

    void removeFocusedObjectBox(QPointF);
};

#endif // LABEL_IMG_H
