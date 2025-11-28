import React, { useState } from 'react';
import { Upload, Button, message, Space, Image } from 'antd';
import { UploadOutlined, CloseOutlined } from '@ant-design/icons';
import type { UploadProps } from 'antd/es/upload';
import './ImageUploader.css';

interface ImageUploaderProps {
  onUpload: (images: string[]) => void;
  onCancel: () => void;
  maxImages?: number;
  maxSize?: number; // in MB
}

const ImageUploader: React.FC<ImageUploaderProps> = ({
  onUpload,
  onCancel,
  maxImages = 5,
  maxSize = 10, // 10MB
}) => {
  const [previewImages, setPreviewImages] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);

  const beforeUpload: UploadProps['beforeUpload'] = (file) => {
    const isImage = file.type.startsWith('image/');
    if (!isImage) {
      message.error('只能上传图片文件！');
      return false;
    }

    const isLtMaxSize = file.size / 1024 / 1024 < maxSize;
    if (!isLtMaxSize) {
      message.error(`图片大小不能超过 ${maxSize}MB！`);
      return false;
    }

    if (previewImages.length >= maxImages) {
      message.error(`最多只能上传 ${maxImages} 张图片！`);
      return false;
    }

    return true;
  };

  const handleFileChange: UploadProps['onChange'] = (info) => {
    if (info.file.status === 'uploading') {
      setUploading(true);
    } else if (info.file.status === 'done') {
      setUploading(false);
      message.success(`${info.file.name} 上传成功`);
    } else if (info.file.status === 'error') {
      setUploading(false);
      message.error(`${info.file.name} 上传失败`);
    }
  };

  const customRequest: UploadProps['customRequest'] = async ({ file, onSuccess, onError }) => {
    try {
      // Convert file to base64
      const reader = new FileReader();
      reader.onload = (e) => {
        const base64 = e.target?.result as string;
        setPreviewImages(prev => [...prev, base64]);
        onSuccess?.(base64);
      };
      reader.onerror = () => {
        onError?.(new Error('Failed to read file'));
      };
      reader.readAsDataURL(file as File);
    } catch (error) {
      onError?.(error as Error);
    }
  };

  const removeImage = (index: number) => {
    setPreviewImages(prev => prev.filter((_, i) => i !== index));
  };

  const handleConfirm = () => {
    if (previewImages.length === 0) {
      message.warning('请先上传图片');
      return;
    }
    onUpload(previewImages);
  };

  const uploadProps: UploadProps = {
    multiple: true,
    showUploadList: false,
    beforeUpload,
    onChange: handleFileChange,
    customRequest,
    accept: 'image/*',
    disabled: previewImages.length >= maxImages,
  };

  return (
    <div className="image-uploader">
      <div className="uploader-content">
        <div className="upload-area">
          <Upload {...uploadProps}>
            <Button
              type="dashed"
              icon={<UploadOutlined />}
              disabled={previewImages.length >= maxImages}
              style={{ width: '100%', height: '120px' }}
            >
              <div>
                <p className="ant-upload-text">点击或拖拽上传图片</p>
                <p className="ant-upload-hint">
                  支持单张或批量上传，最多 {maxImages} 张，每张不超过 {maxSize}MB
                </p>
              </div>
            </Button>
          </Upload>
        </div>

        {previewImages.length > 0 && (
          <div className="preview-area">
            <div className="preview-title">预览 ({previewImages.length}/{maxImages})</div>
            <div className="preview-images">
              {previewImages.map((image, index) => (
                <div key={index} className="preview-image-wrapper">
                  <Image
                    src={image}
                    alt={`预览图片 ${index + 1}`}
                    className="preview-image"
                    preview={{
                      mask: (
                        <div className="image-actions">
                          <Button
                            type="text"
                            icon={<CloseOutlined />}
                            onClick={(e) => {
                              e.stopPropagation();
                              removeImage(index);
                            }}
                            danger
                          />
                        </div>
                      ),
                    }}
                  />
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="uploader-actions">
          <Space>
            <Button onClick={onCancel}>取消</Button>
            <Button
              type="primary"
              onClick={handleConfirm}
              disabled={previewImages.length === 0}
              loading={uploading}
            >
              确认上传 ({previewImages.length})
            </Button>
          </Space>
        </div>
      </div>
    </div>
  );
};

export default ImageUploader;