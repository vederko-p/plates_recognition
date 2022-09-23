
class FakeModel:
    def __call__(self, frame):
        result = {
            'number': '123',
            'color': 'red',
            'type': 'big',
            'timestamp': 'day'
        }
        return result
