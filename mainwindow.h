#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QWheelEvent>
#include <QTableWidgetItem>
#include <QMessageBox>
#include <QRectF>

#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <opencv2/dnn/dnn.hpp>

#include "labelimage.h"


namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_pushButton_open_files_clicked();
    void on_pushButton_change_dir_clicked();
    void on_pushButton_save_clicked();
    void on_pushButton_remove_clicked();

    void on_pushButton_prev_clicked();
    void on_pushButton_next_clicked();

    void keyPressEvent(QKeyEvent *);

    void next_img(bool bSavePrev = true);
    void prev_img(bool bSavePrev = true);
    void save_label_data() const;
    void clear_label_data();
    void remove_img();

    void next_label();
    void prev_label();

    void on_tableWidget_label_cellDoubleClicked(int , int );
    void on_tableWidget_label_cellClicked(int , int );

    void on_horizontalSlider_images_sliderMoved(int );

private:
    void            init();
    void            init_table_widget();
    void            init_button_event();
    void            init_horizontal_slider();

    void            set_label_progress(const int);
    void            set_focused_file(const int);

    void            goto_img(const int);

    void            load_label_list_data(QString);
    QString         get_labeling_data(QString)const;

    void            set_label(const int);
    void            set_label_color(const int , const QColor);

    void            pjreddie_style_msgBox(QMessageBox::Icon, QString, QString);

    bool            open_video_file();
    bool            open_img_dir();
    bool            open_obj_file();

    void            reupdate_img_list();

    Ui::MainWindow *ui;

protected:
    void    wheelEvent(QWheelEvent *);

private:
    QString         m_imgDir;
    QStringList     m_imgList;
    int             m_imgIndex;

    cv::Ptr<cv::VideoCapture> m_video;
    cv::Mat m_currentCvFrame;

    QStringList     m_objList;
    int             m_objIndex;

    cv::dnn::Net    m_net;

    QVector<ObjectLabelingBox> _detectObjects(const cv::Mat & image);
};

#endif // MAINWINDOW_H
