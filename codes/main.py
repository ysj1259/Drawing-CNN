import pygame
import numpy as np
from scenes import *

pygame.init()
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
size = [750, 500]
screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()

opening = opening_scene(screen)
scene2 = scene2(screen)
scene3 = scene3(screen)
scene3_check = scene3_check(screen)
scene4 = scene4(screen)
scene5 = scene5(screen)
done = False
scene1_bool = True
scene2_bool = False
scene3_bool = False
scene3c_bool = False
scene4_bool = False
scene5_bool = False

#scene1
image_start = pygame.image.load('../images/Start.png')
image_start = pygame.transform.scale(image_start,(image_start.get_rect().size[0] / 3, image_start.get_rect().size[1] / 3))
image_start_rect = image_start.get_rect()

#scene2
image_datas = pygame.image.load('../images/datas.png')
image_datas = pygame.transform.scale(image_datas, (image_datas.get_rect().size[0] /4 , image_datas.get_rect().size[1] /4))
image_datas_rect = image_datas.get_rect()

image_arrow1 = pygame.image.load('../images/arrow.png')
image_arrow1 = pygame.transform.scale(image_arrow1, (image_arrow1.get_rect().size[0] / 10, image_arrow1.get_rect().size[1] / 10))
image_arrow1_rect = image_arrow1.get_rect()
image_arrow2 = pygame.image.load('../images/arrow.png')
image_arrow2 = pygame.transform.scale(image_arrow2, (image_arrow2.get_rect().size[0] / 10, image_arrow2.get_rect().size[1] / 10))
image_arrow2_rect = image_arrow2.get_rect()
image_arrow3 = pygame.image.load('../images/arrow.png')
image_arrow3 = pygame.transform.scale(image_arrow3, (image_arrow3.get_rect().size[0] / 10, image_arrow3.get_rect().size[1] / 10))
image_arrow3_rect = image_arrow3.get_rect()

image_layers1 = pygame.image.load('../images/layers.png')
image_layers1 = pygame.transform.scale(image_layers1, (image_layers1.get_rect().size[0] / 8, image_layers1.get_rect().size[1] / 8))
image_layers1_rect = image_layers1.get_rect()
image_layers2 = pygame.image.load('../images/layers.png')
image_layers2 = pygame.transform.scale(image_layers2, (image_layers2.get_rect().size[0] / 8, image_layers2.get_rect().size[1] / 8))
image_layers2_rect = image_layers2.get_rect()

image_connected = pygame.image.load('../images/connected.png')
image_connected = pygame.transform.scale(image_connected, (image_connected.get_rect().size[0] / 8, image_connected.get_rect().size[0] / 8))
image_connected_rect = image_connected.get_rect()

#scene3

#scene4
learning_bool = False
return_num = 0

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

        if (scene1_bool):
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()
                    if image_start_rect.collidepoint(mouse_pos):
                        print('Scene2로 넘어갑니다...')
                        scene1_bool = False
                        scene2_bool = True

        if (scene2_bool):
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    print('Scene3로 넘어갑니다...')
                    scene2_bool = False
                    scene3_bool = True

        if (scene3_bool):
            mouse_pressed = pygame.mouse.get_pressed()
            if (mouse_pressed[0]):
                mouse_pos = pygame.mouse.get_pos()
                scene3.change_cell(mouse_pos)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    scene3.go_next()
                    if (scene3.get_dataidx == 90):
                        print("scene3 check로 넘어갑니다..")
                        scene3_bool = False
                        scene3c_bool = True

        if (scene3c_bool):
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    scene3_check.go_next()
                    if (scene3_check.get_dataidx == 90):
                        print('scene4로 넘어갑니다..')
                        scene3c_bool = False
                        scene4_bool = True

        if (scene4_bool):
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    if (return_num == 1):
                        print('scene5로 넘어갑니다..')
                        scene5_bool = True
                        scene4_bool = False
                        return_num = -1
                    return_num += 1

        if (scene5_bool):
            mouse_pressed = pygame.mouse.get_pressed()
            if (mouse_pressed[0]):
                mouse_pos = pygame.mouse.get_pos()
                scene5.change_cell(mouse_pos)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    if (return_num > 0):
                        scene5.cell_reset()
                        print('cell 초기화')
                    return_num += 1

    # scene 1
    if (scene1_bool and not scene2_bool):
        image_start_rect.center = opening.opening(image_start, image_start_rect)

    # scene 2
    if (not scene1_bool and scene2_bool and not scene3_bool):
        scene2.scene2(image_datas, image_datas_rect, image_arrow1, image_arrow1_rect, image_layers1, image_layers1_rect, image_arrow2, image_arrow2_rect, image_layers2, image_layers2_rect, image_arrow3, image_arrow3_rect, image_connected, image_connected_rect)

    # scene 3
    if (not scene2_bool and scene3_bool and not scene4_bool):
        scene3.text()
        scene3.draw_line()
        scene3.draw_cell()

    if (not scene3_bool and scene3c_bool and not scene4_bool):
        scene3_check.text()
        scene3_check.draw_cell(scene3.get_Xdata)


    if (not scene3c_bool and scene4_bool and not scene5_bool):
        scene4.scene4()
        if (not learning_bool):
            scene4.learning(scene3.get_Xdata, scene3.get_ydata)
            learning_bool = True

    if (not scene4_bool and scene5_bool):
        scene5.text()
        scene5.draw_line()
        scene5.draw_cell()
        scene5.data_predict(scene4.get_model)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
