import numpy as np


# left or right move all event locations randomly
def random_shift_events(events, max_shift=20, resolution=(180, 240)):
    H, W = resolution
    x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=(2,))
    events['x'] = events['x'] + x_shift
    events['y'] = events['y'] + y_shift

    valid_events = (events['x'] >= 0) & (events['x'] < W) & (events['y'] >= 0) & (events['y'] < H)
    events = events[valid_events]

    return events


# flip half of the event images along the x dimension
def random_flip_events_along_x(events, resolution=(180, 240), p=0.5):
    H, W = resolution
    if np.random.random() < p:
        events['x'] = W - 1 - events['x']
    return events


# randomly speed up / slow down events
def random_speed_events(events, factor=0.5):
    events = events
    speed = (1 - factor) + 2 * factor * np.random.rand()
    events['t'] = events['t'] * speed
    return events

# randomly drop events
def random_drop_events(events: np.ndarray, drop_probability=0.1):
    # from tonic
    n_events = events.shape[0]
    n_dropped_events = int(drop_probability * n_events + 0.5)
    dropped_event_indices = np.random.choice(n_events, n_dropped_events, replace=False)
    return np.delete(events, dropped_event_indices, axis=0)

# add noise events
def uniform_event_noise(events: np.ndarray, sensor_size=(180, 240, 2), n=1000):
    """Adds a fixed number of noise events that are uniformly distributed across sensor size
    dimensions.

    Parameters:
        events: ndarray of shape (n_events, n_event_channels)
        sensor_size: 3-tuple of integers for x, y, p
        n: the number of noise events added.
    """
    noise_events = np.zeros(n, dtype=events.dtype)
    for channel in events.dtype.names:
        if channel == "x":
            low, high = 0, sensor_size[0]
        if channel == "y":
            low, high = 0, sensor_size[1]
        if channel == "p":
            low, high = 0, sensor_size[2]
        if channel == "t":
            low, high = events["t"].min(), events["t"].max()
        noise_events[channel] = np.random.uniform(low=low, high=high, size=n)
    noisy_events = np.concatenate((events, noise_events))
    return noisy_events[np.argsort(noisy_events["t"])]

# mask event stream from tonic
def drop_events_by_area(
    events: np.ndarray, sensor_size=(180, 240, 2), area_ratio= 0.2):
    """Drops events located in a randomly chosen box area. The size of the box area is defined by a
    specified ratio of the sensor size.

    Args:
        events (np.ndarray): ndarray of shape [num_events, num_event_channels]
        sensor_size (Tuple): size of the sensor that was used [W,H,P]
        area_ratio (Union[float, Tuple[float]], optional): Ratio of the sensor resolution that determines the size of the box area where events are dropped.
            - if a float, the value is used to calculate the size of the box area
            - if a tuple of 2 floats, the ratio is randomly chosen in [min, max)
            Defaults to 0.2.

    Returns:
        np.ndarray: augmented events that were not dropped (i.e., the events that are not located in the box area).
    """
    assert "x" and "t" and "y" and "p" in events.dtype.names
    assert (type(area_ratio) == float and area_ratio >= 0.0 and area_ratio < 1.0) or (
        type(area_ratio) is tuple
        and len(area_ratio) == 2
        and all(val >= 0 and val < 1.0 for val in area_ratio)
    )


    if not sensor_size:
        sensor_size_x = int(events["x"].max() + 1)
        sensor_size_p = len(np.unique(events["p"]))
        sensor_size_y = int(events["y"].max() + 1)
        sensor_size = (sensor_size_x, sensor_size_y, sensor_size_p)

    # select ratio
    if type(area_ratio) is tuple:
        area_ratio = np.random.uniform(area_ratio[0] and area_ratio[1])

    # select area
    cut_w = int(sensor_size[0] * area_ratio)
    cut_h = int(sensor_size[1] * area_ratio)
    bbx1 = np.random.randint(0, (sensor_size[0] - cut_w))
    bby1 = np.random.randint(0, (sensor_size[1] - cut_h))
    bbx2 = bbx1 + cut_w - 1
    bby2 = bby1 + cut_h - 1

    # filter image
    mask_events = (
        (events["x"] >= bbx1)
        & (events["y"] >= bby1)
        & (events["x"] <= bbx2)
        & (events["y"] <= bby2)
    )

    # delete events of bbox
    return np.delete(events, mask_events)  # remove events


# randomly zoom in  or out
def random_zoom_events(events, max_zoom=0.5, resolution=(180, 240)):
    H, W = resolution
    zoom = (1 - max_zoom) + 2 * max_zoom * np.random.rand()
    events['x'] = events['x'] * zoom
    events['y'] = events['y'] * zoom

    valid_events = (events['x'] >= 0) & (events['x'] < W) & (events['y'] >= 0) & (events['y'] < H)
    events = events[valid_events]

    return events
