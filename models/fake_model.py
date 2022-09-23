
class FakeModel:
    def __call__(self, frame):
        result = {
            'number',
            'color',
            'type',
            'timestamp'
        }
        return result
