
import asyncio
import datetime
import gzip
import json
import time
import uuid
import wave
from io import BytesIO
import aiofiles
import websockets
# pip install websockets==10.4

PROTOCOL_VERSION = 0b0001
DEFAULT_HEADER_SIZE = 0b0001

# Message Type:
FULL_CLIENT_REQUEST = 0b0001
AUDIO_ONLY_REQUEST = 0b0010
FULL_SERVER_RESPONSE = 0b1001
SERVER_ACK = 0b1011
SERVER_ERROR_RESPONSE = 0b1111

# Message Type Specific Flags
NO_SEQUENCE = 0b0000  # no check sequence
POS_SEQUENCE = 0b0001
NEG_SEQUENCE = 0b0010
NEG_WITH_SEQUENCE = 0b0011
NEG_SEQUENCE_1 = 0b0011

# Message Serialization
NO_SERIALIZATION = 0b0000
JSON = 0b0001

# Message Compression
NO_COMPRESSION = 0b0000
GZIP = 0b0001



def generate_header(
        message_type=FULL_CLIENT_REQUEST,
        message_type_specific_flags=NO_SEQUENCE,
        serial_method=JSON,
        compression_type=GZIP,
        reserved_data=0x00):
    """
    protocol_version(4 bits), header_size(4 bits),
    message_type(4 bits), message_type_specific_flags(4 bits)
    serialization_method(4 bits) message_compression(4 bits)
    reserved (8bits)
    """
    header = bytearray()
    header_size = 1
    header.append((PROTOCOL_VERSION << 4) | header_size)
    header.append((message_type << 4) | message_type_specific_flags)
    header.append((serial_method << 4) | compression_type)
    header.append(reserved_data)
    return header


def generate_before_payload(sequence: int):
    before_payload = bytearray()
    before_payload.extend(sequence.to_bytes(4, 'big', signed=True))  # sequence
    return before_payload


def parse_response(res):
    """
    协议头结构  4字节
    [0] protocol_version(4 bits) | header_size(4 bits)
    [1] message_type(4 bits) | message_type_specific_flags(4 bits)
    [2] serialization_method(4 bits) | message_compression(4 bits)
    [3] reserved(8 bits)
    header_extensions 扩展头(大小等于 8 * 4 * (header_size - 1) )
    payload 类似与http 请求体
    """
    # ========== 协议头解析 ==========
    protocol_version = res[0] >> 4
    header_size = res[0] & 0x0f            # 实际头大小 = 值 × 4字节（0b0001→4字节）

    message_type = res[1] >> 4              # 消息类型：0b1001=服务端响应, 0b1011=ACK
    message_type_specific_flags = res[1] & 0x0f # # bit0:含seq, bit1:是否末包

    serialization_method = res[2] >> 4      # 序列化方法：0b0001=JSON
    message_compression = res[2] & 0x0f     # 压缩类型：0b0001=GZIP

    reserved = res[3]                       # 保留字段（当前未使用）

    # ========== 载荷处理 ==========
    header_extensions = res[4:header_size * 4]  # 扩展头（当前规范未使用）
    payload = res[header_size * 4:]         # 实际载荷数据

    result = {'is_last_package': False,}
    payload_msg = None
    payload_size = 0

    # 处理序列号（当 flags 的 bit0 置位时）
    if message_type_specific_flags & 0x01:  # # full cilent 或者 有语音包的时候 (服务端接收客户端)
        # receive frame with sequence
        seq = int.from_bytes(payload[:4], "big", signed=True)
        result['payload_sequence'] = seq
        payload = payload[4:]       # 移除header部分

    # 判断是否末包（当 flags 的 bit1 置位时）
    if message_type_specific_flags & 0x02:  # 最后一个语音包
        # receive last package
        result['is_last_package'] = True

    # 客户端接收服务端 根据不同消息类型处理载荷
    if message_type == FULL_SERVER_RESPONSE:         # 0b1001 语音识别结果
        payload_size = int.from_bytes(payload[:4], "big", signed=True)
        payload_msg = payload[4:]
    elif message_type == SERVER_ACK:            # 0b1011 服务端ACK
        seq = int.from_bytes(payload[:4], "big", signed=True)
        result['seq'] = seq
        if len(payload) >= 8:
            payload_size = int.from_bytes(payload[4:8], "big", signed=False)
            payload_msg = payload[8:]
    elif message_type == SERVER_ERROR_RESPONSE:             # 0b1111 错误响应
        code = int.from_bytes(payload[:4], "big", signed=False)
        result['code'] = code
        payload_size = int.from_bytes(payload[4:8], "big", signed=False)
        payload_msg = payload[8:]

    if payload_msg is None:
        return result

    # 处理压缩数据（与客户端的 gzip.compress 对应）
    if message_compression == GZIP:
        payload_msg = gzip.decompress(payload_msg)
    # 反序列化（与客户端的 json.dumps 对应）
    if serialization_method == JSON:
        payload_msg = json.loads(str(payload_msg, "utf-8"))
    elif serialization_method != NO_SERIALIZATION:
        payload_msg = str(payload_msg, "utf-8")
    result['payload_msg'] = payload_msg
    result['payload_size'] = payload_size
    return result

def read_wav_info(data: bytes = None) -> (int, int, int, int, bytes):
    with BytesIO(data) as _f:
        wave_fp = wave.open(_f, 'rb')
        nchannels, sampwidth, framerate, nframes = wave_fp.getparams()[:4]
        wave_bytes = wave_fp.readframes(nframes)
    return nchannels, sampwidth, framerate, nframes, wave_bytes


def judge_wav(ori_date):
    if len(ori_date) < 44:
        return False
    if ori_date[0:4] == b"RIFF" and ori_date[8:12] == b"WAVE":
        return True
    return False


class AsrWsClient:
    def __init__(self, audio_path, **kwargs):
        """
        :param config: config
        """
        self.audio_path = audio_path
        self.success_code = 1000  # success code, default is 1000
        self.seg_duration = int(kwargs.get("seg_duration", 100))
        self.ws_url = kwargs.get("ws_url", "ws://192.168.1.89:3000/asr")
        self.uid = kwargs.get("uid", "test")
        self.format = kwargs.get("format", "wav")
        self.rate = kwargs.get("rate", 16000)
        self.bits = kwargs.get("bits", 16)
        self.channel = kwargs.get("channel", 1)
        self.codec = kwargs.get("codec", "raw")
        self.auth_method = kwargs.get("auth_method", "none")
        self.hot_words = kwargs.get("hot_words", None)
        self.streaming = kwargs.get("streaming", True)
        self.mp3_seg_size = kwargs.get("mp3_seg_size", 1000)
        self.req_event = 1

    def construct_request(self, reqid, data=None):
        req = {
            "user": {
                "uid": self.uid,
            },
            "audio": {
                'format': self.format,
                "sample_rate": self.rate,
                "bits": self.bits,
                "channel": self.channel,
                "codec": self.codec,
            },
            "request":{
                "model_name": "bigmodel",
                "enable_punc": True,
                # "result_type": "single",
                # "vad_segment_duration": 800,
            }
        }
        return req

    @staticmethod
    def slice_data(data: bytes, chunk_size: int) -> (list, bool):
        data_len = len(data)
        offset = 0
        while offset + chunk_size < data_len:
            yield data[offset: offset + chunk_size], False
            offset += chunk_size
        else:
            yield data[offset: data_len], True


    async def send_data(self, ws, wav_data: bytes, segment_size: int):
        reqid = str(uuid.uuid4())       # 生成一个唯一的请求 ID
        seq = 1
        request_params = self.construct_request(reqid)  # 构建一个请求头
        payload_bytes = str.encode(json.dumps(request_params))   # 转换为 JSON 格式的字符串，然后将其编码为字节
        payload_bytes = gzip.compress(payload_bytes)        # 使用 gzip 压缩算法对负载数据进行压缩
        full_client_request = bytearray(generate_header(message_type_specific_flags=POS_SEQUENCE))
        full_client_request.extend(generate_before_payload(sequence=seq))
        full_client_request.extend((len(payload_bytes)).to_bytes(4, 'big'))  # payload size(4 bytes)
        full_client_request.extend(payload_bytes)  # payload
        await ws.send(full_client_request)
        res = await ws.recv()
        print(ws.response_headers)
        result = parse_response(res)
        print("******************")
        print("sauc result", result)
        print("******************")

        for _, (chunk, last) in enumerate(AsrWsClient.slice_data(wav_data, segment_size), 1):
            seq += 1
            if last:
                seq = -seq
            start = time.time()
            payload_bytes = gzip.compress(chunk)
            audio_only_request = bytearray(generate_header(message_type=AUDIO_ONLY_REQUEST, message_type_specific_flags=POS_SEQUENCE))
            if last:
                audio_only_request = bytearray(generate_header(message_type=AUDIO_ONLY_REQUEST, message_type_specific_flags=NEG_WITH_SEQUENCE))
            audio_only_request.extend(generate_before_payload(sequence=seq))
            audio_only_request.extend((len(payload_bytes)).to_bytes(4, 'big'))  # payload size(4 bytes)
            audio_only_request.extend(payload_bytes)  # payload

            try:
                await ws.send(audio_only_request)
                if self.streaming:
                    sleep_time = max(0, (self.seg_duration / 1000.0 - (time.time() - start)))
                    await asyncio.sleep(sleep_time)
            except Exception as e:
                print(f"Unexpected error: {e}")
                break

    async def receive_data(self, ws):
        result = None
        while True:
            try:
                res = await ws.recv()
                result = parse_response(res)
                print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}, res", result)
                # 末包判断
                if result['payload_sequence'] < 0 or result['is_last_package']:
                    break
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket connection closed")
                break

        return result

    async def segment_data_processor(self, wav_data: bytes, segment_size: int):
        header = {}
        header["X-Api-Resource-Id"] = ""    # 使用资源识别
        header["X-Api-Access-Key"] = "0519"     # 身份验证
        header["X-Api-App-Key"] = ""        # 应用验证
        header["X-Api-Request-Id"] = str(uuid.uuid4())  # 用户标识
        try:
            async with websockets.connect(self.ws_url, extra_headers=header, max_size=1000000000) as ws:
                send_task = asyncio.create_task(self.send_data(ws, wav_data, segment_size))
                receive_task = asyncio.create_task(self.receive_data(ws))

                results = await asyncio.gather(send_task, receive_task)
                send_result, receive_result = results

        except websockets.exceptions.ConnectionClosedError as e:
            print(f"WebSocket connection closed with status code: {e.code}")
            print(f"WebSocket connection closed with reason: {e.reason}")
        except websockets.exceptions.WebSocketException as e:
            print(f"WebSocket connection failed: {e}")
            if hasattr(e, "status_code"):
                print(f"Response status code: {e.status_code}")
            if hasattr(e, "headers"):
                print(f"Response headers: {e.headers}")
            if hasattr(e, "response") and hasattr(e.response, "text"):
                print(f"Response body: {e.response.text}")
        except Exception as e:
            print(f"Unexpected error: {e}")

        return receive_result

    async def execute(self):
        async with aiofiles.open(self.audio_path, mode="rb") as _f:
            data = await _f.read()
        audio_data = bytes(data)
        if self.format == "mp3":
            segment_size = self.mp3_seg_size
            return await self.segment_data_processor(audio_data, segment_size)
        if self.format == "wav":
            nchannels, sampwidth, framerate, nframes, wav_len = read_wav_info(audio_data)
            size_per_sec = nchannels * sampwidth * framerate
            segment_size = int(size_per_sec * self.seg_duration / 1000)
            return await self.segment_data_processor(audio_data, segment_size)
        if self.format == "pcm":
            segment_size = int(self.rate * 2 * self.channel * self.seg_duration / 500)
            return await self.segment_data_processor(audio_data, segment_size)
        else:
            raise Exception("Unsupported format")


def execute_one(audio_item, **kwargs):
    assert 'id' in audio_item
    assert 'path' in audio_item
    audio_id = audio_item['id']
    audio_path = audio_item['path']
    asr_http_client = AsrWsClient(
        audio_path=audio_path,
        **kwargs
    )
    result = asyncio.run(asr_http_client.execute())
    return {"id": audio_id, "path": audio_path, "result": result}


def test_stream():
    print("测试流式")
    result = execute_one(
        {
            'id': 1,
            "path": "vad_example.wav"
        }
    )
    print(result)


if __name__ == '__main__':
    test_stream()
"""
conda activate fireredasr
python simplex_websocket_demo.py
"""