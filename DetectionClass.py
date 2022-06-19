from test2 import *
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import numpy as np
from matplotlib import pyplot as plt
import cv2
import myfunctions2 as mf2
import copy

# 1. Create Paths
class DetectionClass:
    #def __init__(self):
    ###Algorithm###
    TOMATO_MIN_PRECISION = 0.7
    STEM_MIN_PRECISION = 0.05

    STEM_MAX_DISTANCE = 1.5
    # Stem power
    SP_DISTANCE = 10
    SP_ANGLE = 5
    SP_PRECISION = 2
    SP_TOMATO_PRECISION = 5
    STEM_FINAL_MIN_PRECISION = 0
    # Debug
    TUNING = False
    ######
    pomCheck = 15
    szyCheck = 7

    paths = {
        'APIMODEL_PATH': os.path.join('Tensorflow', 'models'),
        'IMAGE_PATH': os.path.join('Images'),
        'CHECKPOINT_PATH': os.path.join('Pomidor'),
        'CHECKPOINT_PATH_SZY': os.path.join('Szypulka'),
    }

    files = {
        'LABELMAP': os.path.join('Pomidor', 'label_map.pbtxt'),
        'LABELMAP_SZY': os.path.join('Szypulka', 'label_map.pbtxt'),
        'PIPELINE_CONFIG': os.path.join('Pomidor', 'pipeline.config'),
        'PIPELINE_CONFIG_SZY': os.path.join('Szypulka', 'pipeline.config')
    }

# 2. Create Label Map
    labels = [{'name': 'Pomidor', 'id': 2}]
    with open(files['LABELMAP'], 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')

    labels2 = [{'name': 'Szypulka', 'id': 1}]
    with open(files['LABELMAP_SZY'], 'w') as f:
        for label in labels2:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-'+str(pomCheck))).expect_partial()

    # Load pipeline config and build a detection model (Szypułka)
    configs_szy = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG_SZY'])
    detection_model_szy = model_builder.build(model_config=configs_szy['model'], is_training=False)

    # Restore checkpoint (Szypułka)
    ckpt_szy = tf.compat.v2.train.Checkpoint(model=detection_model_szy)
    ckpt_szy.restore(os.path.join(paths['CHECKPOINT_PATH_SZY'], 'ckpt-'+str(szyCheck))).expect_partial()

    category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
    category_index_szy = label_map_util.create_category_index_from_labelmap(files['LABELMAP_SZY'])

    @tf.function
    def detect_fn(self, image, detection_model_local):
        image, shapes = detection_model_local.preprocess(image)
        prediction_dict = detection_model_local.predict(image, shapes)
        detections = detection_model_local.postprocess(prediction_dict, shapes)
        return detections

    @tf.function
    def detect_fn2(self, image, detection_model_local):
        image, shapes = detection_model_local.preprocess(image)
        prediction_dict = detection_model_local.predict(image, shapes)
        detections = detection_model_local.postprocess(prediction_dict, shapes)
        return detections

    def calc_value(self, obj, tomato):
        if obj[4] < self.STEM_MIN_PRECISION:
            return 0
        if mf2.dist(mf2.get_center(tomato), mf2.get_center(obj)) / mf2.get_radius(tomato) > self.STEM_MAX_DISTANCE:
            return 0
        #print(tomato[4])
        #print(mf2.dist(mf2.get_center(tomato), mf2.get_center(obj)))
        #print(mf2.get_radius(tomato))
        #print(mf2.dist(mf2.get_center(tomato), mf2.get_center(obj))/mf2.get_radius(tomato))
        val_d = mf2.dist(mf2.get_center(tomato),
                         mf2.get_center(obj))  # get_radius(tomato) / (dist(get_center(tomato), get_center(obj)))
        val_a = mf2.angle(obj, tomato)
        value = (val_a ** self.SP_ANGLE) * (val_d ** (1/self.SP_DISTANCE)) * (obj[4] ** self.SP_PRECISION) * (
                tomato[4] ** self.SP_TOMATO_PRECISION)
        #print(value)
        return value

    def detection(self, DEBUG = False, file = 0):
        #IMAGE_PATH = os.path.join(self.paths['IMAGE_PATH'], 'tom_gr_8_Color.png')

        #tmp = camera_view()
        if file != 0:
            img = cv2.imread(file)
        else:
            img = cv2.imread('input.png')  ################################################
        height, width, channels = img.shape
        image_np = np.array(img)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor, self.detection_model)
        detections_szy = self.detect_fn2(input_tensor, self.detection_model_szy)

        ###
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        ###
        num_detections_szy = int(detections_szy.pop('num_detections'))
        detections_szy = {key: value[0, :num_detections_szy].numpy()
                          for key, value in detections_szy.items()}
        detections_szy['num_detections'] = num_detections_szy

        # detection_classes should be ints.
        detections_szy['detection_classes'] = detections_szy['detection_classes'].astype(np.int64)

        tomatos = []
        for i in range(10):
            arr_score = detections['detection_scores']
            arr_boxes = detections['detection_boxes']
            if arr_score[i] >= self.TOMATO_MIN_PRECISION:
                tomatos.append([arr_boxes[i][0], arr_boxes[i][1], arr_boxes[i][2], arr_boxes[i][3], arr_score[i]])
            else:
                break

        stems = []
        for i, ele in enumerate(detections_szy['detection_scores']):
            arr_score = detections_szy['detection_scores']
            arr_boxes = detections_szy['detection_boxes']
            stems.append([arr_boxes[i][0], arr_boxes[i][1], arr_boxes[i][2], arr_boxes[i][3], arr_score[i]])

        # print(tomatos)
        if (DEBUG):
            print(stems)

        tomatos_stems_list = []
        selected_detected_stems = []

        for i, tomato in enumerate(tomatos):
            # image = cv2.imread(path)
            best_stem = [0, 0, 0, 0, 0, 0]
            stem_list = []
            for stem in stems:
                local_stem = copy.copy(stem)
                local_stem.append(self.calc_value(local_stem, tomato))
                if (local_stem[5] > best_stem[5]):
                    best_stem = local_stem
                # print(local_stem)
                stem_list.append(local_stem)

            # if best_stem[5] >= STEM_FINAL_PRECISION:
            #  selected_detected_stems.append(copy.copy(best_stem))

            stem_list.sort(reverse=True, key=lambda x: x[5])
            # print()
            # print(best_stem)
            tomatos_stems_list.append(stem_list[0:len(tomatos)])
            #print()

        # print(tomatos_stems_list)
        for counter in range(len(tomatos)):
            best_stem_index = 0
            best_stem_score = 0
            for i, stem_list in enumerate(tomatos_stems_list):
                best_score = stem_list[0][5]  # max(stem_list, key=lambda x: x[5])[5]
                if best_score > best_stem_score:
                    best_stem_score = best_score
                    best_stem_index = i
                    #print(f"Changed = {best_stem_score}")
            # print(best_score)
            best_stem_list = tomatos_stems_list[best_stem_index]
            tomatos_stems_list.remove(best_stem_list)
            best = best_stem_list[0]
            if (DEBUG):
                print(best_stem_list[0])
            # print(counter)
            for i, stem_list in enumerate(tomatos_stems_list):
                #print(len(tomatos_stems_list[i]))
                filter(lambda x: x[0:4] == best[0:4], tomatos_stems_list[i])
                for stem in stem_list:
                    if stem[0:4] == best[0:4]:
                        stem_list.remove(stem)
                #print(len(tomatos_stems_list[i]))

            #print(best[5])
            if best[5] > self.STEM_FINAL_MIN_PRECISION:
                selected_detected_stems.append(best)

        if self.TUNING:
            ds = np.asarray([stem[5] for stem in selected_detected_stems])  # for algorithm tuning
        else:
            ds = np.asarray([stem[4] for stem in selected_detected_stems])
        dc = np.asarray([0 for stem in selected_detected_stems])
        db = np.asarray([box[0:4] for box in selected_detected_stems])

        #print(f"tomatos = {tomatos}")
        #print(f"db = {db}")

        detections_stem = {'detection_scores': ds, 'detection_classes': dc.astype(int), 'detection_boxes': db}

        # In[ ]:

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        # print(detections['detection_boxes'])
        # print(len(detections['detection_boxes']))
        # print(len(detections['detection_classes']))
        # print(detections['detection_classes'])
        # print(detections['detection_scores'])

        # Final
        # tomatos
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset + 1,
            detections['detection_scores'],
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=100,
            min_score_thresh=self.TOMATO_MIN_PRECISION,
            agnostic_mode=False)
        # stems
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections_stem['detection_boxes'],
            detections_stem['detection_classes'] + label_id_offset,
            detections_stem['detection_scores'],
            self.category_index_szy,
            use_normalized_coordinates=True,
            max_boxes_to_draw=100,
            min_score_thresh=0,
            agnostic_mode=False)

        #plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
        #plt.show()

        cv2.imwrite("out.png", image_np_with_detections)

        ###Debug###
        # tomatos
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset + 1,
            detections['detection_scores'],
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=100,
            min_score_thresh=self.TOMATO_MIN_PRECISION,
            agnostic_mode=False)
        # stems
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections_szy['detection_boxes'],
            detections_szy['detection_classes'] + label_id_offset,
            detections_szy['detection_scores'],
            self.category_index_szy,
            use_normalized_coordinates=True,
            max_boxes_to_draw=100,
            min_score_thresh=self.STEM_MIN_PRECISION,
            agnostic_mode=False)

        #plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
        #plt.show()

        cv2.imwrite("out_debug.png", image_np_with_detections)

        return (tomatos[0:4], db[0:4], width, height)

    def real_time_detection(self):
        cap = cv2.VideoCapture(0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while cap.isOpened():
            ret, frame = cap.read()
            image_np = np.array(frame)

            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = self.detect_fn(input_tensor, self.detection_model)
            detections_szy = self.detect_fn2(input_tensor, self.detection_model_szy)

            ###
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            ###
            num_detections_szy = int(detections_szy.pop('num_detections'))
            detections_szy = {key: value[0, :num_detections_szy].numpy()
                              for key, value in detections_szy.items()}
            detections_szy['num_detections'] = num_detections_szy

            # detection_classes should be ints.
            detections_szy['detection_classes'] = detections_szy['detection_classes'].astype(np.int64)

            tomatos = []
            for i in range(10):
                arr_score = detections['detection_scores']
                arr_boxes = detections['detection_boxes']
                if arr_score[i] >= self.TOMATO_MIN_PRECISION:
                    tomatos.append([arr_boxes[i][0], arr_boxes[i][1], arr_boxes[i][2], arr_boxes[i][3], arr_score[i]])
                else:
                    break

            stems = []
            for i, ele in enumerate(detections_szy['detection_scores']):
                arr_score = detections_szy['detection_scores']
                arr_boxes = detections_szy['detection_boxes']
                stems.append([arr_boxes[i][0], arr_boxes[i][1], arr_boxes[i][2], arr_boxes[i][3], arr_score[i]])

            # print(tomatos)
            #if (DEBUG):
             #   print(stems)

            tomatos_stems_list = []
            selected_detected_stems = []

            for i, tomato in enumerate(tomatos):
                # image = cv2.imread(path)
                best_stem = [0, 0, 0, 0, 0, 0]
                stem_list = []
                for stem in stems:
                    local_stem = copy.copy(stem)
                    local_stem.append(self.calc_value(local_stem, tomato))
                    if (local_stem[5] > best_stem[5]):
                        best_stem = local_stem
                    # print(local_stem)
                    stem_list.append(local_stem)

                # if best_stem[5] >= STEM_FINAL_PRECISION:
                #  selected_detected_stems.append(copy.copy(best_stem))

                stem_list.sort(reverse=True, key=lambda x: x[5])
                #print(stem_list)
                # print(best_stem)
                tomatos_stems_list.append(stem_list[0:len(tomatos)])
                # print()

            # print(tomatos_stems_list)
            for counter in range(len(tomatos)):
                best_stem_index = 0
                best_stem_score = 0
                for i, stem_list in enumerate(tomatos_stems_list):
                    best_score = stem_list[0][5]  # max(stem_list, key=lambda x: x[5])[5]
                    if best_score > best_stem_score:
                        best_stem_score = best_score
                        best_stem_index = i
                        # print(f"Changed = {best_stem_score}")
                # print(best_score)
                best_stem_list = tomatos_stems_list[best_stem_index]
                tomatos_stems_list.remove(best_stem_list)
                best = best_stem_list[0]
                #if (DEBUG):
                 #   print(best_stem_list[0])
                # print(counter)
                for i, stem_list in enumerate(tomatos_stems_list):
                    # print(len(tomatos_stems_list[i]))
                    filter(lambda x: x[0:4] == best[0:4], tomatos_stems_list[i])
                    for stem in stem_list:
                        if stem[0:4] == best[0:4]:
                            stem_list.remove(stem)
                    # print(len(tomatos_stems_list[i]))
                selected_detected_stems.append(best)

            if self.TUNING:
                ds = np.asarray([stem[5] for stem in selected_detected_stems])  # for algorithm tuning
            else:
                ds = np.asarray([stem[4] for stem in selected_detected_stems])
            dc = np.asarray([0 for stem in selected_detected_stems])
            db = np.asarray([box[0:4] for box in selected_detected_stems])

            # print(f"tomatos = {tomatos}")
            # print(f"db = {db}")

            detections_stem = {'detection_scores': ds, 'detection_classes': dc.astype(int), 'detection_boxes': db}

            # In[ ]:

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            # print(detections['detection_boxes'])
            # print(len(detections['detection_boxes']))
            # print(len(detections['detection_classes']))
            # print(detections['detection_classes'])
            # print(detections['detection_scores'])

            # Final
            # tomatos
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'] + label_id_offset + 1,
                detections['detection_scores'],
                self.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=100,
                min_score_thresh=self.TOMATO_MIN_PRECISION,
                agnostic_mode=False)
            # stems
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections_stem['detection_boxes'],
                detections_stem['detection_classes'] + label_id_offset,
                detections_stem['detection_scores'],
                self.category_index_szy,
                use_normalized_coordinates=True,
                max_boxes_to_draw=100,
                min_score_thresh=0,
                agnostic_mode=False)

            # plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
            # plt.show()

            #cv2.imwrite("out.png", image_np_with_detections)

            cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
