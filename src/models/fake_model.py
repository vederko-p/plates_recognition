
class FakeModel:
    def __call__(self, frame):
        result = {
            'number': '123',  # RUS letters
            'color': 'red',  # HTML HEX
            'type': 'big',
            'timestamp': 'day'  # h-m-s
        }
        return result
