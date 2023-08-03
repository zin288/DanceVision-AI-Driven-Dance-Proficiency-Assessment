import cv2
from abc import ABC, abstractmethod
from enum import Enum
import uuid
import imutils


def read_image(image_path, width, height, mask_transparent):
    if mask_transparent:
        _image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # make mask of where the transparent bits are
        trans_mask = _image[:, :, 3] == 0

        # replace areas of transparency with white and not transparent
        _image[trans_mask] = [228, 169, 0, 255]
        # new image without alpha channel...
        _image = cv2.cvtColor(_image, cv2.COLOR_BGRA2BGR)
    else:
        _image = cv2.imread(image_path)
    _image = imutils.resize(_image, width, height)
    return _image


class Shape(ABC):
    class Meta:
        class State(Enum):
            INACTIVE = 0
            ENTER = 1
            EXIT = 2

    def __init__(self, anchor_x, anchor_y, label):
        """

        :param anchor_x: anchor x-point for shape.  Could be upper left or center
        :type anchor_x:
        :param anchor_y: anchor y-point for shape.  Could be upper left or center
        :type anchor_y:
        """
        self.anchor_x = anchor_x
        self.anchor_y = anchor_y
        self.label = label
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.state = Shape.Meta.State.INACTIVE
        self.id = uuid.uuid4()

    # https://gist.github.com/xcsrz/8938a5d4a47976c745407fe2788c813a
    def _center_text(self, text):
        # get boundary of this text
        textsize = cv2.getTextSize(text, self.font, 1, 2)[0]
        # get coords based on boundary
        textX = self.anchor_x + (textsize[0] // 2)
        textY = self.anchor_y + (textsize[1] // 2)

        return textX, textY

    @abstractmethod
    def is_point_inside(self, x, y):
        pass

    @abstractmethod
    def process_point(self, x, y, image, event=None):
        pass

    @abstractmethod
    def on_enter(self, x, y, image):
        pass

    @abstractmethod
    def on_exit(self, x, y, image):
        pass

    @abstractmethod
    def on_click(self, x, y, image):
        pass

    @abstractmethod
    def draw(self, image):
        pass


class ButtonShape(Shape, ABC):
    def __init__(self, x, y, label):
        super().__init__(int(x), int(y), label)
        self.click_cb = None

    def set_click_callback(self, click_callback):
        self.click_cb = click_callback

    def on_click(self, x, y, image):
        if self.click_cb is not None:
            self.click_cb()


class RectButton(ButtonShape):
    def _center_text(self, text):
        # get boundary of this text
        (w, h), baseline = cv2.getTextSize(text, self.font, 1, 2)

        self.width = int(w * 1.1)
        self.height = int(h * 1.1) + baseline

        # get coords based on boundary
        textX = self.anchor_x + (self.width - w) // 2
        textY = self.anchor_y + (self.height - h) // 2 + h

        self.X_text = textX
        self.Y_text = textY

    def __init__(self, x, y, label, inactive_background_color, active_background_color, outline_color):
        super().__init__(int(x), int(y), label)
        self._center_text(label)
        self.outline_color = outline_color
        self.inactive_bkg_color = inactive_background_color
        self.active_bkg_color = active_background_color
        self.line_thickness = -1

    def draw(self, image):
        if self.state == Shape.Meta.State.INACTIVE:
            cv2.rectangle(image, pt1=(self.anchor_x, self.anchor_y),
                          pt2=(self.anchor_x + self.width, self.anchor_y + self.height), color=self.inactive_bkg_color,
                          thickness=self.line_thickness)
        else:
            cv2.rectangle(image, pt1=(self.anchor_x, self.anchor_y),
                          pt2=(self.anchor_x + self.width, self.anchor_y + self.height), color=self.active_bkg_color,
                          thickness=self.line_thickness)

        # add text centered on image, (textX,textY) is the BOTTOM LEFT
        cv2.putText(image, self.label, (self.X_text, self.Y_text), self.font, 1, (255, 255, 255), 2)

    def on_enter(self, x, y, image):
        pass

    def on_exit(self, x, y, image):
        pass

    def process_point(self, x, y, image, event=None):
        in_shape = self.is_point_inside(x, y)
        if self.state is Shape.Meta.State.INACTIVE:
            if in_shape == True:
                self.state = Shape.Meta.State.ENTER
                self.on_enter(x, y, image)
        elif self.state == Shape.Meta.State.ENTER:
            if not in_shape:
                self.state = Shape.Meta.State.EXIT
                self.on_exit(x, y, image)
                self.state = Shape.Meta.State.INACTIVE

        if in_shape and event == cv2.EVENT_LBUTTONDOWN:
            self.on_click(x, y, image)

    def is_point_inside(self, x, y):
        in_shape = self.anchor_x < x < self.anchor_x + self.width and self.anchor_y < y < self.anchor_y + self.height
        return in_shape


class CircleButton(ButtonShape):

    def __init__(self, x, y, radius, label, outline_color):
        super().__init__(int(x), int(y), label)
        self.radius = radius
        self.outline_color = outline_color

    def draw(self, image):
        if self.state == Shape.Meta.State.INACTIVE:
            cv2.circle(image, (self.anchor_x, self.anchor_y), self.radius, self.outline_color, 2, lineType=cv2.LINE_AA)
            # cv2.circle(image, (self.anchor_x, self.anchor_y), int(self.radius*0.8), self.outline_color, -1)

            textX, textY = self._center_text(self.label)
            # add text centered on image
            cv2.putText(image, self.label, (textX, textY + self.radius + 15), self.font, 1, (255, 255, 255), 2)
        else:
            cv2.circle(image, (self.anchor_x, self.anchor_y), self.radius, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            cv2.circle(image, (self.anchor_x, self.anchor_y), int(self.radius * 0.4), self.outline_color, -1)
            textX, textY = self._center_text(self.label)
            # add text centered on image
            cv2.putText(image, self.label, (textX, textY + self.radius + 15), self.font, 1, (255, 255, 255), 2)

    def on_enter(self, x, y, image):
        pass

    def on_exit(self, x, y, image):
        pass

    def on_click(self, x, y, image):
        pass

    def process_point(self, x, y, image, event=None):
        in_circle = self.is_point_inside(x, y)
        if self.state is Shape.Meta.State.INACTIVE:
            if in_circle == True:
                self.state = Shape.Meta.State.ENTER
                self.on_enter(x, y, image)
        elif self.state == Shape.Meta.State.ENTER:
            if not in_circle:
                self.state = Shape.Meta.State.EXIT
                self.on_exit(x, y, image)
                self.state = Shape.Meta.State.INACTIVE

    def is_point_inside(self, x, y):
        in_circle = (x - self.anchor_x) ** 2 + (y - self.anchor_y) ** 2 < self.radius ** 2
        return in_circle


class DisplayValueLabel(Shape):
    def __init__(self, x, y, width, height, label, bkgnd_color=(245, 117, 16), value_color=(255, 255, 255),
                 label_value_space=10):
        super().__init__(x, y, label)
        self.width = width
        self.height = height
        self.bkgnd_color = bkgnd_color
        # textsize = tuple = (x,y)
        self.textsize = cv2.getTextSize(self.label, self.font, 1, 2)[0]
        self.value = None
        self.value_color = value_color
        self.label_x = self.anchor_x + 10
        self.label_y = self.anchor_y + 25
        self.value_x = self.anchor_x + self.textsize[0] + label_value_space
        self.value_y = self.anchor_y + 25

    def set_value(self, val):
        self.value = val

    def draw(self, image):
        cv2.rectangle(image, (self.anchor_x, self.anchor_y), (self.anchor_x + self.width, self.anchor_y + self.height),
                      self.bkgnd_color, -1)

        # Display Class
        cv2.putText(image, self.label
                    , (self.label_x, self.label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f"{self.value}"
                    , (self.value_x, self.value_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    def on_enter(self, x, y, image):
        pass

    def on_exit(self, x, y, image):
        pass

    def on_click(self, x, y, image):
        pass

    def process_point(self, x, y, image, event=None):
        pass

    def is_point_inside(self, x, y):
        return False


class RectangleHotSpot(Shape):
    """
    A hotspot is an invisible rectangular area.  This class will notify the user on mouse events
    or (x,y) values are enter, exit, click to process, etc

    """

    def __init__(self, rect, label=""):
        """
        :param rect: (ul-x, ul-y, lr-x, lr-y)
        :type rect: tuple
        """
        self.rect = rect
        super().__init__(rect[0], rect[1], label)

    def _rectContains(self, pt_x, pt_y):
        """
        :param rect: (ix,iy,x,y)
        :type rect:
        :param pt: (new x,new y)
        :type pt:
        :return:
        :rtype:
        """
        logic = self.rect[0] < pt_x < self.rect[2] and self.rect[1] < pt_y < self.rect[3]
        return logic

    def process_point(self, x, y, image, event=None):
        pass

    def on_enter(self, x, y, image):
        if self.state != self.Meta.State.ENTER:
            if self._rectContains(x, y):
                self.state = self.Meta.State.ENTER
                return True
        else:
            return False

    def on_exit(self, x, y, image):
        if self.state == self.Meta.State.ENTER:
            if not self._rectContains(x, y):
                self.state = self.Meta.State.INACTIVE
                return True
        else:
            return False

    def on_click(self, x, y, image):
        pass

    def draw(self, image):
        # invisible hotspot
        pass

    def is_point_inside(self, x, y):
        return self._rectContains(x, y)


class SolidColorRect(RectangleHotSpot):

    def __init__(self, rect, colors=[(255, 255, 255)], label=""):
        """
        :param rect: (ul-x, ul-y, lr-x, lr-y)
        :type rect: tuple
        """
        self.rect = rect
        self.colors = colors
        self.color_index = 0
        super().__init__(rect, label)

    def reset_color_index(self):
        self.color_index=0

    def process_point(self, x, y, image, event=None):
        self.color_index += 1
        if self.color_index >= len(self.colors):
            self.color_index = 0

    def draw(self, image, color=None):
        if color is None:
            color = self.colors[self.color_index]
        cv2.rectangle(image, pt1=(self.rect[0], self.rect[1]), pt2=(self.rect[2], self.rect[3]), color=color,
                      thickness=-1)
