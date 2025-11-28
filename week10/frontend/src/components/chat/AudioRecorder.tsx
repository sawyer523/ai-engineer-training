import React, { useState, useRef, useEffect } from 'react';
import { Button, message } from 'antd';
import { 
  AudioOutlined, 
  PauseOutlined, 
  CheckOutlined, 
  CloseOutlined,
  SoundOutlined,
  DeleteOutlined
} from '@ant-design/icons';
import './AudioRecorder.css';

interface AudioRecorderProps {
  onRecord: (audioData: string, duration: number) => void;
  onCancel: () => void;
  maxDuration?: number; // in seconds
}

interface AudioRecorderState {
  isRecording: boolean;
  isPaused: boolean;
  recordingTime: number;
  audioBlob: Blob | null;
  audioUrl: string | null;
  volume: number;
}

const AudioRecorder: React.FC<AudioRecorderProps> = ({ 
  onRecord, 
  onCancel,
  maxDuration = 60 // 1 minute max
}) => {
  const [state, setState] = useState<AudioRecorderState>({
    isRecording: false,
    isPaused: false,
    recordingTime: 0,
    audioBlob: null,
    audioUrl: null,
    volume: 0,
  });

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const recordingIntervalRef = useRef<number | null>(null);
  const volumeIntervalRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      // Cleanup on unmount
      stopAll();
    };
  }, []);

  const stopAll = () => {
    if (recordingIntervalRef.current) {
      clearInterval(recordingIntervalRef.current);
      recordingIntervalRef.current = null;
    }
    if (volumeIntervalRef.current) {
      clearInterval(volumeIntervalRef.current);
      volumeIntervalRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
  };

  const setupAudioContext = (stream: MediaStream) => {
    try {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;
      source.connect(analyserRef.current);
    } catch (error) {
      console.warn('Audio context setup failed:', error);
    }
  };

  const getVolumeLevel = (): number => {
    if (!analyserRef.current) return 0;
    
    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteFrequencyData(dataArray);
    
    const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;
    return Math.min(100, (average / 255) * 100);
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100,
        } 
      });
      
      streamRef.current = stream;
      setupAudioContext(stream);

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
      });
      
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm;codecs=opus' });
        const url = URL.createObjectURL(blob);
        
        setState(prev => ({
          ...prev,
          audioBlob: blob,
          audioUrl: url,
          isRecording: false,
          isPaused: false,
        }));
        
        stopAll();
      };

      mediaRecorder.start(100); // Collect data every 100ms
      
      setState(prev => ({ ...prev, isRecording: true, isPaused: false }));

      // Start recording timer
      recordingIntervalRef.current = setInterval(() => {
        setState(prev => {
          const newTime = prev.recordingTime + 1;
          if (newTime >= maxDuration) {
            stopRecording();
            return prev;
          }
          return { ...prev, recordingTime: newTime };
        });
      }, 1000);

      // Start volume monitoring
      volumeIntervalRef.current = setInterval(() => {
        setState(prev => ({ ...prev, volume: getVolumeLevel() }));
      }, 100);

    } catch (error) {
      console.error('Failed to start recording:', error);
      message.error('无法访问麦克风，请检查权限设置');
      stopAll();
    }
  };

  const pauseRecording = () => {
    if (mediaRecorderRef.current && state.isRecording && !state.isPaused) {
      mediaRecorderRef.current.pause();
      setState(prev => ({ ...prev, isPaused: true }));
      
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current);
      }
      if (volumeIntervalRef.current) {
        clearInterval(volumeIntervalRef.current);
      }
    }
  };

  const resumeRecording = () => {
    if (mediaRecorderRef.current && state.isRecording && state.isPaused) {
      mediaRecorderRef.current.resume();
      setState(prev => ({ ...prev, isPaused: false }));
      
      // Restart timers
      recordingIntervalRef.current = setInterval(() => {
        setState(prev => {
          const newTime = prev.recordingTime + 1;
          if (newTime >= maxDuration) {
            stopRecording();
            return prev;
          }
          return { ...prev, recordingTime: newTime };
        });
      }, 1000);
      
      volumeIntervalRef.current = setInterval(() => {
        setState(prev => ({ ...prev, volume: getVolumeLevel() }));
      }, 100);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && state.isRecording) {
      mediaRecorderRef.current.stop();
    }
  };

  const deleteRecording = () => {
    stopAll();
    if (state.audioUrl) {
      URL.revokeObjectURL(state.audioUrl);
    }
    setState({
      isRecording: false,
      isPaused: false,
      recordingTime: 0,
      audioBlob: null,
      audioUrl: null,
      volume: 0,
    });
  };

  const confirmRecording = () => {
    if (state.audioBlob && state.audioUrl) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64Audio = reader.result as string;
        onRecord(base64Audio, state.recordingTime);
        URL.revokeObjectURL(state.audioUrl!);
      };
      reader.readAsDataURL(state.audioBlob);
    }
  };

  const cancelRecording = () => {
    stopAll();
    if (state.audioUrl) {
      URL.revokeObjectURL(state.audioUrl);
    }
    onCancel();
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const getVolumeBarColor = () => {
    if (state.volume < 30) return '#52c41a'; // Green
    if (state.volume < 70) return '#faad14'; // Yellow
    return '#ff4d4f'; // Red
  };

  const renderRecordingControls = () => (
    <div className="recording-controls">
      {!state.isRecording ? (
        <Button
          type="primary"
          icon={<AudioOutlined />}
          onClick={startRecording}
          size="large"
        >
          开始录音
        </Button>
      ) : (
        <>
          {state.isPaused ? (
            <Button
              type="primary"
              icon={<AudioOutlined />}
              onClick={resumeRecording}
              size="large"
            >
              继续录音
            </Button>
          ) : (
            <Button
              type="default"
              icon={<PauseOutlined />}
              onClick={pauseRecording}
              size="large"
            >
              暂停录音
            </Button>
          )}
          
          <Button
            type="primary"
            danger
            icon={<CloseOutlined />}
            onClick={stopRecording}
            size="large"
          >
            停止录音
          </Button>
        </>
      )}
      
      <Button
        type="text"
        icon={<CloseOutlined />}
        onClick={cancelRecording}
        size="large"
      >
        取消
      </Button>
    </div>
  );

  const renderPlaybackControls = () => (
    <div className="playback-controls">
      <div className="playback-info">
        <audio controls src={state.audioUrl!} className="playback-audio" />
        <p className="playback-duration">时长：{formatTime(state.recordingTime)}</p>
      </div>

      <div className="playback-actions">
        <Button
          type="primary"
          icon={<CheckOutlined />}
          onClick={confirmRecording}
          size="large"
        >
          确认发送
        </Button>
        
        <Button
          type="default"
          icon={<DeleteOutlined />}
          onClick={deleteRecording}
          size="large"
        >
          重新录制
        </Button>
        
        <Button
          type="text"
          icon={<CloseOutlined />}
          onClick={cancelRecording}
          size="large"
        >
          取消
        </Button>
      </div>
    </div>
  );

  return (
    <div className="audio-recorder">
      <div className="recorder-content">
        {!state.audioBlob ? (
          <>
            <div className="recording-indicator">
              {state.isRecording && (
                <div className="recording-status">
                  <div className="recording-pulse">
                    <div className="pulse-dot" style={{ 
                      backgroundColor: getVolumeBarColor(),
                      transform: `scale(${1 + state.volume / 200})`
                    }}></div>
                    <span className="recording-time">{formatTime(state.recordingTime)}</span>
                  </div>
                  
                  <div className="volume-indicator">
                    <SoundOutlined style={{ marginRight: 8 }} />
                    <div className="volume-bar">
                      <div 
                        className="volume-fill" 
                        style={{ 
                          width: `${state.volume}%`,
                          backgroundColor: getVolumeBarColor()
                        }}
                      />
                    </div>
                    <span className="volume-text">{Math.round(state.volume)}%</span>
                  </div>
                </div>
              )}
              
              {!state.isRecording && (
                <div className="recording-ready">
                  <AudioOutlined style={{ fontSize: '32px', color: '#1890ff', marginBottom: '8px' }} />
                  <h3>语音输入</h3>
                  <p>点击开始录音，最多可录制 {maxDuration} 秒</p>
                </div>
              )}
            </div>

            {renderRecordingControls()}
          </>
        ) : (
          renderPlaybackControls()
        )}
      </div>
    </div>
  );
};

export default AudioRecorder;