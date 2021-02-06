classes = ['Speed limit (20km/h)',
           'Speed limit (30km/h)',
           'No passing veh over 3.5 tons',
           'Right-of-way at intersection',
           'Priority road',
           'Yield',
           'Stop',
           'No vehicles',
           'Vehicle > 3.5 tons prohibited',
           'No entry',
           'General caution',
           'Dangerous curve left',
           'Speed limit (50km/h)',
           'Dangerous curve right',
           'Double curve',
           'Bumpy road',
           'Slippery road',
           'Road narrows on the right',
           'Road work',
           'Traffic signals',
           'Pedestrians',
           'Children crossing',
           'Bicycles crossing',
           'Speed limit (60km/h)',
           'Beware of ice/snow',
           'Wild animals crossing',
           'End speed + passing limits',
           'Turn right ahead',
           'Turn left ahead',
           'Ahead only',
           'Go straight or right',
           'Go straight or left',
           'Keep right',
           'Keep left',
           'Speed limit (70km/h)',
           'Roundabout mandatory',
           'End of no passing',
           'End no passing vehicle > 3.5 tons',
           'Speed limit (80km/h)',
           'End of speed limit (80km/h)',
           'Speed limit (100km/h)',
           'Speed limit (120km/h)',
           'No passing']


def label(i):
    #! Classes of trafic sign
    return classes[i]


if __name__ == "__main__":
    print(label(2))
