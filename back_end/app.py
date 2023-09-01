from flask_socketio import SocketIO
import flask
from flask import Flask, request
from functions.detect_img import *
from functions.track_video import *
from functions.track_video_player import *
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, supports_credentials=True, resource={r'/*': {'origins': '*'}})
socketio = SocketIO(app)
# 设置保存检测结果图片的目录
image_result_path = "./results/images/results.jpg"

video_result_path = "./results/videos/results.mp4"


@app.route('/trackByImg', methods=['POST'])
@cross_origin()
def track_by_img():
    """
        --检测图片--
        收到前端返回图片文件类型 调用检测
        :return:检测后结果图片
        """
    file = request.files.get("file")
    file__ = request.form['suffix']
    path = 'results\images.' + file__
    file.save(path)
    if file is not None:
        detect_img(path)
        return flask.send_file(image_result_path, as_attachment=True)


@app.route('/trackByVideo', methods=['POST'])
def process_video():
    # 获取前端发送的视频数据
    file = request.files.get('file')
    track_type = request.form['trackType'].split(',')
    Model = request.form['model'].split(',')
    if Model[0] == '0':
        Model = 'botsort'
    else:
        Model = 'bytetrack'
    file__ = request.form['suffix']
    path = 'results\Videos.' + file__
    file.save(path)
    if file is not None:
        if track_type[0] == 'Retrograde detection':
            track_video(path, Model)
            return flask.send_file(video_result_path, as_attachment=True)
        elif track_type[0] == 'Player trajectory tracking':
            track_video_player(path, Model)
            return flask.send_file(video_result_path, as_attachment=True)


if __name__ == '__main__':
    socketio.run(app, host="localhost", debug=True, port=5000, allow_unsafe_werkzeug=True)
