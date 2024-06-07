import pygame
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy import *

class default():
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

class opening_scene(default):
    def __init__(self, screen):
        self.screen = screen


    def opening(self, image, image_rect):
        self.screen.fill(default.WHITE)
        font = pygame.font.SysFont('malgungothic', 30)
        text1 = font.render(" CNN 알고리즘 체험하기: ", True, default.BLACK)
        text2 = font.render("숫자 예측 알고리즘 제작 시뮬레이션", True, default.BLACK)
        self.screen.blit(text1, (200, 100))
        self.screen.blit(text2, (130, 200))
        self.screen.blit(image, (self.screen.get_size()[0] / 2 - image_rect.size[0] / 2, 300))
        return (self.screen.get_size()[0] / 2, 300 + image_rect.size[1] / 2)

class scene2(default):

    def __init__(self, screen):
        self.screen = screen

    def scene2(self, image1, image_rect1, image2, image_rect2, image3, image_rect3, image4, image_rect4, image5, image_rect5, image6, image_rect6, image7, image_rect7):
        self.screen.fill(default.WHITE)

        image_rect1.center = (100, 250)
        self.screen.blit(image1, image_rect1.topleft)
        image_rect2.center = (220, 250)
        self.screen.blit(image2, image_rect2.topleft)
        image_rect3.center = (300, 250)
        self.screen.blit(image3, image_rect3.topleft)
        image_rect4.center = (390, 250)
        self.screen.blit(image4, image_rect4.topleft)
        image_rect5.center = (470, 250)
        self.screen.blit(image5, image_rect5.topleft)
        image_rect6.center = (560, 250)
        self.screen.blit(image6, image_rect6.topleft)
        image_rect7.center = (640, 250)
        self.screen.blit(image7, image_rect7.topleft)

        font = pygame.font.SysFont('malgungothic', 20)
        text1 = font.render("우리가 사용할 모델 구성은 다음과 같습니다.", True, default.BLACK)
        self.screen.blit(text1, (50, 50))
        font2 = pygame.font.SysFont('malgungothic', 15)
        text2 = font2.render("layer: 50 * 50 * 1", True, default.BLACK)
        self.screen.blit(text2, (50, 310))
        text3 = font2.render("convolution layer", True, default.BLACK)
        self.screen.blit(text3, (240, 310))
        text4 = font2.render("layer: 48 * 48 * 64", True, default.BLACK)
        self.screen.blit(text4, (240, 330))
        font3 = pygame.font.SysFont('malgungothic', 12)
        text5 = font3.render("Max pooling: 25 * 25 * 64", True, default.BLACK)
        self.screen.blit(text5, (230, 350))
        text6 = font2.render("convolution layer", True, default.BLACK)
        self.screen.blit(text6, (420, 310))
        text7 = font2.render("layer: 20 * 20 * 128", True, default.BLACK)
        self.screen.blit(text7, (410, 330))
        text8 = font3.render("Max pooling: 10 * 10 * 128", True, default.BLACK)
        self.screen.blit(text8, (400, 350))
        text9 = font2.render("connected layer", True, default.BLACK)
        self.screen.blit(text9, (590, 310))
        text10 = font2.render("layer: 1 * 12800", True, default.BLACK)
        self.screen.blit(text10, (590, 330))
        text11 = font2.render("enter을 누르면 다음으로 넘어갑니다...", True, default.BLACK)
        self.screen.blit(text11, (10, 470))

class scene3(default):
    def __init__(self, screen):
        self.screen = screen
        self._X_data = []
        self._y_data = []
        self._cells = np.zeros((50, 50, 1), dtype = int)
        self._data_idx = 0
    def text(self):
        self.screen.fill(default.WHITE)
        number = (self._data_idx // 10) + 1
        idx = (self._data_idx % 10) + 1
        font = pygame.font.SysFont('malgungothic', 20)
        text1 = font.render(f"학습할 {number}의 사진을", True, default.BLACK)
        text2 = font.render("그려주세요", True, default.BLACK)
        text3 = font.render(f"{idx} / 10", True, default.BLACK)
        self.screen.blit(text1, (20, 50))
        self.screen.blit(text2, (20, 70))
        self.screen.blit(text3, (20, 90))

    def draw_line(self):
        x_pos = np.arange(250, 750, 10)
        y_pos = np.arange(0, 500, 10)
        for i in range(len(x_pos)):
            pygame.draw.line(self.screen, default.BLACK, [x_pos[i], 0], [x_pos[i], 500], width = 1)
            pygame.draw.line(self.screen, default.BLACK, [250, y_pos[i]], [750, y_pos[i]], width = 1)

    def change_cell(self, mouse_pos):
        current_pos = [ ((mouse_pos[0] - 250 )// 10), (mouse_pos[1] // 10)]
        self._cells[current_pos[0]][current_pos[1]][0] = 200

    def draw_cell(self):
        for x in range(50):
            for y in range(50):
                pygame.draw.rect(self.screen, (self._cells[x][y][0], self._cells[x][y][0], self._cells[x][y][0]), (250 + x * 10, y * 10, 10, 10))


    def go_next(self):
        self._X_data.append(self._cells)
        self._y_data.append( (self._data_idx // 10))
        self._cells = np.zeros((50, 50, 1), dtype= int)
        self._data_idx += 1

    @property
    def get_Xdata(self):
        return self._X_data

    @property
    def get_ydata(self):
        return self._y_data

    @property
    def get_dataidx(self):
        return self._data_idx

class scene3_check(scene3):
    def __init__(self, screen):
        self.screen = screen
        self._data_idx = 0

    def draw_cell(self, x_data):
        for x in range(50):
            for y in range(50):
                pygame.draw.rect(self.screen, (x_data[self._data_idx][x][y][0], x_data[self._data_idx][x][y][0], x_data[self._data_idx][x][y][0]),(250 + x * 10, y * 10, 10, 10))

    def text(self):
        super().text()

    def go_next(self):
        self._data_idx += 1

    @property
    def get_dataidx(self):
        return self._data_idx

class scene4(default):
    def __init__(self, screen):
        self.screen = screen
    def scene4(self):
        self.screen.fill(default.WHITE)
        font = pygame.font.SysFont('malgungothic', 50)
        text = font.render("학습중 입니다...", True, default.BLACK)
        self.screen.blit(text, (100, 100))

    def learning(self, X_data, y_data):
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        X_data = X_data.astype('float32') / 255
        y_data = keras.utils.to_categorical(y_data)

        gen = ImageDataGenerator(rotation_range = 20, shear_range = 0.2, width_shift_range = 0.2, height_shift_range = 0.2, horizontal_flip = False)
        augment_ratio = 9
        augment_size = int(augment_ratio * X_data.shape[0])
        randidx = np.random.randint(X_data.shape[0], size = augment_size)

        X_aug = X_data[randidx].copy()
        y_aug = y_data[randidx].copy()
        X_aug, y_aug = next(gen.flow(X_aug, y_aug, batch_size = augment_size, shuffle = True))

        X_train = np.concatenate((X_data, X_aug))
        y_train = np.concatenate((y_data, y_aug))

        self.model = Sequential()
        self.model.add(Conv2D(64, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu', input_shape = (50, 50, 1)))
        self.model.add(MaxPooling2D(pool_size = (2, 2)))
        self.model.add(Conv2D(128, kernel_size = (6, 6), activation = 'relu', padding = 'valid'))
        self.model.add(MaxPooling2D(pool_size = (2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(1000, activation = 'relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(9, activation = 'softmax'))
        self.model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        self.model.fit(X_train, y_train, batch_size = 50, epochs = 20, verbose = 2)
        print('leraning conplete')

    @property
    def get_model(self):
        return self.model


class scene5(scene3):
    def __init__(self, screen):
        self._cells = np.zeros((50, 50, 1), dtype = int)
        self.screen = screen
        self.prediction = 0
    def change_cell(self, mouse_pos):
        super().change_cell(mouse_pos)

    def draw_cell(self):
        super().draw_cell()

    def data_predict(self, model):
        input = self._cells.reshape(1, 50, 50, 1)
        input = input.astype(float) / 255

        self.prediction = model.predict(input)

    def cell_reset(self):
        self._cells = np.zeros((50, 50, 1), dtype = int)

    def text(self):
        self.screen.fill(default.WHITE)
        font = pygame.font.SysFont('malgungothic', 20)
        text1 = font.render(f"데이터 예측 숫자는", True, default.BLACK)
        text2 = font.render(f"{np.argmax(self.prediction) + 1} 입니다.", True, default.BLACK)
        text3 = font.render(f"{np.max(self.prediction):.2%} 입니다.", True, default.BLACK)
        self.screen.blit(text1, (20, 50))
        self.screen.blit(text2, (20, 70))
        self.screen.blit(text3, (20, 90))
