#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>
#include <QColorDialog>
#include <QKeyEvent>
#include <QDebug>
#include <QShortcut>
#include <QFileInfo>
#include <QDir>

#include <iomanip>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    m_settings("APP", "YOLO_LABEL")
{
    ui->setupUi(this);

    connect(new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_S), this), SIGNAL(activated()), this, SLOT(save_label_data()));
    connect(new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_C), this), SIGNAL(activated()), this, SLOT(clear_label_data()));

    connect(new QShortcut(QKeySequence(Qt::Key_S), this), SIGNAL(activated()), this, SLOT(next_label()));
    connect(new QShortcut(QKeySequence(Qt::Key_W), this), SIGNAL(activated()), this, SLOT(prev_label()));
    connect(new QShortcut(QKeySequence(Qt::Key_A), this), SIGNAL(activated()), this, SLOT(prev_img()));
    connect(new QShortcut(QKeySequence(Qt::Key_D), this), SIGNAL(activated()), this, SLOT(next_img()));
    connect(new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_D), this), SIGNAL(activated()), this, SLOT(remove_img()));

    connect(new QShortcut(QKeySequence(Qt::Key_Down), this), SIGNAL(activated()), this, SLOT(next_label()));
    connect(new QShortcut(QKeySequence(Qt::Key_Up), this), SIGNAL(activated()), this, SLOT(prev_label()));
    connect(new QShortcut(QKeySequence(Qt::Key_Left), this), SIGNAL(activated()), this, SLOT(prev_img()));
    connect(new QShortcut(QKeySequence(Qt::Key_Right), this), SIGNAL(activated()), this, SLOT(next_img()));
    connect(new QShortcut(QKeySequence(Qt::Key_Space), this), SIGNAL(activated()), this, SLOT(next_img()));

    init_table_widget();

    m_net = cv::dnn::readNetFromDarknet("D:\\hands\\yolov4-tiny-hand.cfg",
                                        "D:\\hands\\yolov4-tiny-hand_final.weights");
    m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_open_files_clicked()
{
    if (!open_video_file())
        return;

    if (!open_obj_file())
        return;

    init();
}

void MainWindow::on_pushButton_change_dir_clicked()
{
    if (!open_video_file())
        return;

    init();
}


void MainWindow::init()
{
    ui->label_image->init();

    init_button_event();
    init_horizontal_slider();

    set_label(0);
    goto_img(0);
}

void MainWindow::set_label_progress(const int fileIndex)
{
    QString strCurFileIndex = QString::number(fileIndex);
    QString strEndFileIndex;
    if (m_video)
        strEndFileIndex = QString::number(m_video->get(cv::CAP_PROP_FRAME_COUNT) - 1);
    else
        strEndFileIndex = QString::number(m_imgList.size() - 1);

    ui->label_progress->setText(strCurFileIndex + " / " + strEndFileIndex);
}

void MainWindow::set_focused_file(const int fileIndex)
{
    if (m_video)
        ui->label_file->setText(QString("File: video[%1]").arg(fileIndex));
    else
        ui->label_file->setText("File: " + m_imgList.at(fileIndex));
}

void MainWindow::goto_img(const int frameIndex)
{
    if (m_video)
    {
        if (m_imgIndex < 0)
            m_imgIndex = 0;
        if (!m_video->set(cv::CAP_PROP_POS_FRAMES, frameIndex))
        {
            m_imgIndex = 0;
            qWarning() << "Error on seek to frame " << frameIndex;
            return;
        }

        m_imgIndex = frameIndex;

        if (!m_video->read(m_currentCvFrame))
        {
            //m_imgIndex = 0;
            //qWarning() << "Error on seek to frame " << frameIndex;
            return;
        }
        cv::cvtColor(m_currentCvFrame, m_currentCvFrame, cv::COLOR_BGR2RGB);
        ui->label_image->setCvImage(m_currentCvFrame);
        ui->label_image->setDetectedObjects(_detectObjects(m_currentCvFrame));

        ui->label_image->loadLabelData(get_labeling_data(QString(m_imgDir + "/frame_%1.").arg(frameIndex)));
        ui->label_image->showImage();
    }
    else
    {
        bool bIndexIsOutOfRange = (frameIndex < 0 || frameIndex > m_imgList.size() - 1);
        if (bIndexIsOutOfRange) return;

        m_imgIndex = frameIndex;

        ui->label_image->openImage(m_imgList.at(m_imgIndex));

        ui->label_image->loadLabelData(get_labeling_data(m_imgList.at(m_imgIndex)));
        ui->label_image->showImage();
    }

    set_label_progress(m_imgIndex);
    set_focused_file(m_imgIndex);

    //it blocks crash with slider change
    ui->horizontalSlider_images->blockSignals(true);
    ui->horizontalSlider_images->setValue(m_imgIndex);
    ui->horizontalSlider_images->blockSignals(false);
}

void MainWindow::next_img(bool bSavePrev)
{
    if (bSavePrev && ui->label_image->isOpened())
        save_label_data();
    goto_img(m_imgIndex + 1);
}

void MainWindow::prev_img(bool bSavePrev)
{
    if (bSavePrev)
        save_label_data();
    goto_img(m_imgIndex - 1);
}

void MainWindow::save_label_data()const
{
    QString qstrOutputLabelData;

    if (m_video)
    {
        if (ui->label_image->objBoundingBoxes().size() == 0)
            return;
        cv::Mat cvFrame;
        cv::cvtColor(m_currentCvFrame, cvFrame, cv::COLOR_BGR2RGB);
        QString path = m_imgDir + QString("/frame_%1.").arg(m_imgIndex);
        qstrOutputLabelData = get_labeling_data(path);
        cv::imwrite((path + "jpg").toStdString(), cvFrame);
    }
    else
    {
        if (m_imgList.size() == 0)
            return;
        qstrOutputLabelData = get_labeling_data(m_imgList.at(m_imgIndex));
    }

    ofstream fileOutputLabelData(qstrOutputLabelData.toStdString());

    if (fileOutputLabelData.is_open())
    {
        QVector<ObjectLabelingBox> objBoundingBoxes = ui->label_image->objBoundingBoxes();
        for (int i = 0; i < objBoundingBoxes.size(); i++)
        {
            ObjectLabelingBox objBox = objBoundingBoxes[i];

            if (!m_video && ui->checkBox_cropping->isChecked())
            {
                QImage cropped = ui->label_image->crop(ui->label_image->cvtRelativeToAbsoluteRectInImage(objBox.box));

                if (!cropped.isNull())
                {
                    string strImgFile = m_imgList.at(m_imgIndex).toStdString();

                    strImgFile = strImgFile.substr( strImgFile.find_last_of('/') + 1,
                                                    strImgFile.find_last_of('.') - strImgFile.find_last_of('/') - 1);

                    cropped.save(QString().fromStdString(strImgFile) + "_cropped_" + QString::number(i) + ".png");
                }
            }

            double midX     = objBox.box.x() + objBox.box.width() / 2.;
            double midY     = objBox.box.y() + objBox.box.height() / 2.;
            double width    = objBox.box.width();
            double height   = objBox.box.height();

            fileOutputLabelData << objBox.label;
            fileOutputLabelData << " ";
            fileOutputLabelData << std::fixed << std::setprecision(6) << midX;
            fileOutputLabelData << " ";
            fileOutputLabelData << std::fixed << std::setprecision(6) << midY;
            fileOutputLabelData << " ";
            fileOutputLabelData << std::fixed << std::setprecision(6) << width;
            fileOutputLabelData << " ";
            fileOutputLabelData << std::fixed << std::setprecision(6) << height << std::endl;
        }

        fileOutputLabelData.close();

        ui->textEdit_log->setText(qstrOutputLabelData + " saved.");
    }
}

void MainWindow::clear_label_data()
{
    ui->label_image->resetObjBoundingBoxes();
    ui->label_image->showImage();
}

void MainWindow::remove_img()
{
    if (m_video)
    {
        QString labelDataPath = get_labeling_data(m_imgDir + QString("/frame_%1.").arg(m_imgDir).arg(m_imgIndex));
        QFile::remove(labelDataPath);
        m_imgIndex--;
        if (m_imgIndex < 0)
            m_imgIndex = 0;
        goto_img(m_imgIndex);
    }
    else
    {
        if (m_imgList.size() > 0)
        {
            //remove a image
            QFile::remove(m_imgList.at(m_imgIndex));

            //remove a txt file
            QString qstrOutputLabelData = get_labeling_data(m_imgList.at(m_imgIndex));
            QFile::remove(qstrOutputLabelData);

            m_imgList.removeAt(m_imgIndex);

            if (m_imgList.size() == 0)
            {
                pjreddie_style_msgBox(QMessageBox::Information,"End", "In directory, there are not any image. program quit.");
                QCoreApplication::quit();
            }
            else if (m_imgIndex == m_imgList.size())
            {
                m_imgIndex--;
            }

            goto_img(m_imgIndex);
        }
    }
}

void MainWindow::next_label()
{
    set_label(m_objIndex + 1);
}

void MainWindow::prev_label()
{
    set_label(m_objIndex - 1);
}

void MainWindow::load_label_list_data(QString qstrLabelListFile)
{
    ifstream inputLabelListFile(qstrLabelListFile.toStdString());

    if (inputLabelListFile.is_open())
    {

        for (int i = 0 ; i <= ui->tableWidget_label->rowCount(); i++)
            ui->tableWidget_label->removeRow(ui->tableWidget_label->currentRow());

        m_objList.clear();

        ui->tableWidget_label->setRowCount(0);
        QVector<QColor> drawObjectBoxColors;

        string strLabel;
        int fileIndex = 0;
        while (getline(inputLabelListFile, strLabel))
        {
            int nRow = ui->tableWidget_label->rowCount();
  
            QString qstrLabel   = QString().fromStdString(strLabel);
            QColor  labelColor  = LabelImage::BOX_COLORS[(fileIndex++)%10];
            m_objList << qstrLabel;

            ui->tableWidget_label->insertRow(nRow);

            ui->tableWidget_label->setItem(nRow, 0, new QTableWidgetItem(qstrLabel));
            ui->tableWidget_label->item(nRow, 0)->setFlags(ui->tableWidget_label->item(nRow, 0)->flags() ^  Qt::ItemIsEditable);

            ui->tableWidget_label->setItem(nRow, 1, new QTableWidgetItem(QString().fromStdString("")));
            ui->tableWidget_label->item(nRow, 1)->setBackgroundColor(labelColor);
            ui->tableWidget_label->item(nRow, 1)->setFlags(ui->tableWidget_label->item(nRow, 1)->flags() ^  Qt::ItemIsEditable ^  Qt::ItemIsSelectable);

            drawObjectBoxColors.push_back(labelColor);
        }

        ui->label_image->setDrawObjectBoxColors(drawObjectBoxColors);
    }
}

QString MainWindow::get_labeling_data(QString qstrImgFile)const
{
    string strImgFile = qstrImgFile.toStdString();
    string strLabelData = strImgFile.substr(0, strImgFile.find_last_of('.')) + ".txt";

    return QString().fromStdString(strLabelData);
}

void MainWindow::set_label(const int labelIndex)
{
    bool bIndexIsOutOfRange = (labelIndex < 0 || labelIndex > m_objList.size() - 1);

    if (!bIndexIsOutOfRange)
    {
        m_objIndex = labelIndex;
        ui->label_image->setFocusObjectLabel(m_objIndex);
        ui->label_image->setFocusObjectName(m_objList.at(m_objIndex));
        ui->tableWidget_label->setCurrentCell(m_objIndex, 0);
    }
}

void MainWindow::set_label_color(const int index, const QColor color)
{
    QVector<QColor> drawObjectBoxColors = ui->label_image->getDrawObjectBoxColors();
    drawObjectBoxColors.replace(index, color);
    ui->label_image->setDrawObjectBoxColors(drawObjectBoxColors);
}

void MainWindow::pjreddie_style_msgBox(QMessageBox::Icon icon, QString title, QString content)
{
    QMessageBox msgBox(icon, title, content, QMessageBox::Ok);

    QFont font;
    font.setBold(true);
    msgBox.setFont(font);
    msgBox.button(QMessageBox::Ok)->setFont(font);
    msgBox.button(QMessageBox::Ok)->setStyleSheet("border-style: outset; border-width: 2px; border-color: rgb(0, 255, 0); color : rgb(0, 255, 0);");
    msgBox.button(QMessageBox::Ok)->setFocusPolicy(Qt::ClickFocus);
    msgBox.setStyleSheet("background-color : rgb(34, 0, 85); color : rgb(0, 255, 0);");

    msgBox.exec();
}

bool MainWindow::open_video_file()
{
    m_video.reset();
    m_imgDir.clear();
    m_imgList.clear();

    //pjreddie_style_msgBox(QMessageBox::Information,"Help", "Step 1. Open Your Data Set Directory");

    QString videoFilePath = m_settings.value("last_video_path", "./").toString();
    QWidget w(this);
    videoFilePath = QFileDialog::getOpenFileName(
                &w,
                tr("Open Dataset Directory"),
                videoFilePath);
    if (videoFilePath.isEmpty())
        return false;
    m_settings.setValue("last_video_path", videoFilePath);

    QFileInfo videoFileInfo(videoFilePath);
    QDir dir = videoFileInfo.absoluteDir();
    if (!dir.exists(videoFileInfo.baseName()))
    {
        if (!dir.mkdir(videoFileInfo.baseName()))
        {
            pjreddie_style_msgBox(QMessageBox::Critical, "Error", "Cannot make dir for images if video file.");
            return false;
        }
    }
    if (!dir.cd(videoFileInfo.baseName()))
    {
        pjreddie_style_msgBox(QMessageBox::Critical, "Error", "Cannot move to dir of images");
        return false;
    }
    m_imgDir = dir.absolutePath();

    m_video = cv::makePtr<cv::VideoCapture>();
    if (!m_video->open(videoFilePath.toStdString()))
    {
        m_video.reset();
        m_imgDir.clear();
        pjreddie_style_msgBox(QMessageBox::Critical, "Error", "Cannot open video file.");
        return false;
    }

    return true;
}

bool MainWindow::open_img_dir()
{
    m_video.reset();

    //pjreddie_style_msgBox(QMessageBox::Information,"Help", "Step 1. Open Your Data Set Directory");

    QString imageDir = m_settings.value("last_image_dir", "./").toString();
    QWidget w(this);
    imageDir = QFileDialog::getExistingDirectory(
                &w,
                tr("Open Dataset Directory"),
                imageDir,QFileDialog::ShowDirsOnly);
    if (imageDir.isEmpty())
        return false;
    m_settings.setValue("last_image_dir", imageDir);

    QDir dir(imageDir);

    QStringList fileList = dir.entryList(
                QStringList() << "*.jpg" << "*.JPG" << "*.png",
                QDir::Files);

    if (fileList.empty())
    {
        pjreddie_style_msgBox(QMessageBox::Critical,"Error", "This folder is empty");
        return false;
    }

    m_imgDir   = imageDir;
    m_imgList  = fileList;

    for (QString& str: m_imgList)
        str = m_imgDir + "/" + str;
    return true;
}

bool MainWindow::open_obj_file()
{
    //pjreddie_style_msgBox(QMessageBox::Information,"Help", "Step 2. Open Your Label List File(*.txt or *.names)");

    QString fileLabelList = m_settings.value("last_file_label_list", "./").toString();
    QWidget w(this);
    fileLabelList = QFileDialog::getOpenFileName(
                &w,
                tr("Open LabelList file"),
                fileLabelList,
                tr("LabelList Files (*.txt *.names)"));
    if (fileLabelList.isEmpty())
        return false;
    m_settings.setValue("last_file_label_list", fileLabelList);

    load_label_list_data(fileLabelList);
    return true;
}

void MainWindow::reupdate_img_list()
{

}

void MainWindow::wheelEvent(QWheelEvent *ev)
{
    if (ev->delta() > 0) // up Wheel
        prev_img();
    else if (ev->delta() < 0) //down Wheel
        next_img();
}

QVector<ObjectLabelingBox> MainWindow::_detectObjects(const cv::Mat & image)
{
    vector<string> labelNames = m_net.getUnconnectedOutLayersNames();

    cv::Mat outputBlob = cv::dnn::blobFromImage(image,
                           1.0 / 256.0, cv::Size(416, 416),
                           cv::Scalar(0.0), true, false);
    m_net.setInput(outputBlob);
    vector<cv::Mat> output;
    m_net.forward(output, labelNames);

    QVector<ObjectLabelingBox> boxes;
    for (size_t i = 0; i < output.size(); ++i)
    {
        cv::Mat out = output.at(i);
        for (int y = 0; y < out.size().height; ++y)
        {
            float * v_ptr = out.ptr<float>(y, 0);
            float confidence = v_ptr[5];
            if (confidence < 0.01f)
                continue;
            float hw = v_ptr[2] * 0.5f;
            float hh = v_ptr[3] * 0.5f;
            boxes.push_back({ 0, QRectF((v_ptr[0] - hw), (v_ptr[1] - hh),
                                        v_ptr[2], v_ptr[3]) });
        }
    }

    return boxes;
}

void MainWindow::on_pushButton_prev_clicked()
{
    prev_img();
}

void MainWindow::on_pushButton_next_clicked()
{
    next_img();
}

void MainWindow::keyPressEvent(QKeyEvent * event)
{
    int     nKey = event->key();

    bool    graveAccentKeyIsPressed    = (nKey == Qt::Key_QuoteLeft);
    bool    numKeyIsPressed            = (nKey >= Qt::Key_0 && nKey <= Qt::Key_9 );

    if (graveAccentKeyIsPressed)
    {
        set_label(0);
    }
    else if (numKeyIsPressed)
    {
        int asciiToNumber = nKey - Qt::Key_0;

        if (asciiToNumber < m_objList.size() )
        {
            set_label(asciiToNumber);
        }
    }
}

void MainWindow::on_pushButton_save_clicked()
{
    save_label_data();
}

void MainWindow::on_pushButton_remove_clicked()
{
    remove_img();
}

void MainWindow::on_tableWidget_label_cellDoubleClicked(int row, int column)
{
    bool bColorAttributeClicked = (column == 1);

    if(bColorAttributeClicked)
    {
        QColor color = QColorDialog::getColor(Qt::white,nullptr,"Set Label Color");
        if(color.isValid())
        {
            set_label_color(row, color);
            ui->tableWidget_label->item(row, 1)->setBackgroundColor(color);
        }
        set_label(row);
        ui->label_image->showImage();
    }
}

void MainWindow::on_tableWidget_label_cellClicked(int row, int column)
{
    set_label(row);
}

void MainWindow::on_horizontalSlider_images_sliderMoved(int position)
{
    goto_img(position);
}

void MainWindow::init_button_event()
{
    ui->pushButton_change_dir->setEnabled(true);
    ui->pushButton_next->setEnabled(true);
    ui->pushButton_prev->setEnabled(true);
    ui->pushButton_save->setEnabled(true);
    ui->pushButton_remove->setEnabled(true);
}

void MainWindow::init_horizontal_slider()
{
    ui->horizontalSlider_images->setEnabled(true);
    if (m_video)
    {
        ui->horizontalSlider_images->setRange(0, m_video->get(cv::CAP_PROP_FRAME_COUNT) - 1);
    }
    else
    {
        ui->horizontalSlider_images->setRange(0, m_imgList.size() - 1);
    }
    ui->horizontalSlider_images->blockSignals(true);
    ui->horizontalSlider_images->setValue(0);
    ui->horizontalSlider_images->blockSignals(false);
}

void MainWindow::init_table_widget()
{
    ui->tableWidget_label->horizontalHeader()->setVisible(true);
    ui->tableWidget_label->horizontalHeader()->setStyleSheet("");
    ui->tableWidget_label->horizontalHeader()->setStretchLastSection(true);

    disconnect(ui->tableWidget_label->horizontalHeader(), SIGNAL(sectionPressed(int)),ui->tableWidget_label, SLOT(selectColumn(int)));
}
