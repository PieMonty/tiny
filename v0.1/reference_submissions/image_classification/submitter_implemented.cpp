/*
Copyright (C) EEMBC(R). All Rights Reserved

All EEMBC Benchmark Software are products of EEMBC and are provided under the
terms of the EEMBC Benchmark License Agreements. The EEMBC Benchmark Software
are proprietary intellectual properties of EEMBC and its Members and is
protected under all applicable laws, including all applicable copyright laws.

If you received this EEMBC Benchmark Software without having a currently
effective EEMBC Benchmark License Agreement, you must discontinue use.

Copyright 2020 The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file reflects a modified version of th_lib from EEMBC. The reporting logic
in th_results is copied from the original in EEMBC.
==============================================================================*/
/// \file
/// \brief C++ implementations of submitter_implemented.h

#include "api/submitter_implemented.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "api/internally_implemented.h"
#include "mbed.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "util/quantization_helpers.h"
#include "util/tf_micro_model_runner.h"
#include "ic/ic_inputs.h"
#include "ic/ic_model_quant_data.h"
#include "ic/ic_model_settings.h"

UnbufferedSerial pc(USBTX, USBRX, 115200);

constexpr int kTensorArenaSize = 100 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

int8_t input_quantized[kIcInputSize];
 
//tflite::MicroModelRunner<int8_t, int8_t, 7> *runner;
tflite::ErrorReporter* error_reporter = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
const tflite::Model* model = nullptr;
TfLiteTensor* model_input = nullptr;

  
// Implement this method to prepare for inference and preprocess inputs.
void th_load_tensor() {
  size_t bytes = ee_get_buffer(reinterpret_cast<uint8_t *>(input_quantized),
                               kIcInputSize * sizeof(uint8_t));
  if (bytes / sizeof(uint8_t) != kIcInputSize) {
    th_printf("Input db has %d elemented, expected %d\n", bytes / sizeof(uint8_t),
              kIcInputSize);
    return;
  }
 
 /*
  //runner->SetInput(input_quantized);
      // Populate input tensor with an image with no person.
    TfLiteTensor* input = interpreter->input(0);
    int8_t* input_buffer = tflite::GetTensorData<int8_t>(input);
    int input_length = input->bytes / sizeof(int8_t);
    for (int i = 0; i < input_length; i++) {
      input_buffer[i] = input_quantized[i];
    }
    * */
}

// Add to this method to return real inference results.
void th_results() {
  const int nresults = 10;
  /**
   * The results need to be printed back in exactly this format; if easier
   * to just modify this loop than copy to results[] above, do that.
   */
  th_printf("m-results-[");
  int kCategoryCount = 10;
  TfLiteTensor* output = interpreter->output(0);
  for (size_t i = 0; i < kCategoryCount; i++) {
    float converted =
        DequantizeInt8ToFloat(output->data.int8[i], interpreter->output(0)->params.scale,
                                              interpreter->output(0)->params.zero_point);
    th_printf("%0.3f", converted);
    if (i < (nresults - 1)) {
      th_printf(",");
    }
  }
  th_printf("]\r\n");
}

// Implement this method with the logic to perform one inference cycle.
void th_infer() { interpreter->Invoke();}//runner->Invoke(); }

/// \brief optional API.
void th_final_initialize(void) {
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  
  model = tflite::GetModel(pretrainedResnet_quant_tflite);
 
  static tflite::MicroMutableOpResolver<7> resolver(error_reporter);
  resolver.AddAdd();
  resolver.AddFullyConnected();
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddAveragePool2D();
 // static tflite::MicroModelRunner<int8_t, int8_t, 7> model_runner(
   //   pretrainedResnet_quant_tflite, resolver, tensor_arena, kTensorArenaSize);
   
   model = tflite::GetModel(pretrainedResnet_quant_tflite);

      	// from runner to interpreter
  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  //runner = &model_runner;
  
    // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }
  
    // Get information about the memory area to use for the model's input.
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 3) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] !=
       (3072)) ||
      (model_input->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad input tensor parameters in model");
    return;
  }
  th_printf("Initialized\r\n");
}
void th_pre() {}
void th_post() {}

void th_command_ready(char volatile *p_command) {
  p_command = p_command;
  ee_serial_command_parser_callback((char *)p_command);
}

// th_libc implementations.
int th_strncmp(const char *str1, const char *str2, size_t n) {
  return strncmp(str1, str2, n);
}

char *th_strncpy(char *dest, const char *src, size_t n) {
  return strncpy(dest, src, n);
}

size_t th_strnlen(const char *str, size_t maxlen) {
  return strnlen(str, maxlen);
}

char *th_strcat(char *dest, const char *src) { return strcat(dest, src); }

char *th_strtok(char *str1, const char *sep) { return strtok(str1, sep); }

int th_atoi(const char *str) { return atoi(str); }

void *th_memset(void *b, int c, size_t len) { return memset(b, c, len); }

void *th_memcpy(void *dst, const void *src, size_t n) {
  return memcpy(dst, src, n);
}

/* N.B.: Many embedded *printf SDKs do not support all format specifiers. */
int th_vprintf(const char *format, va_list ap) { return vprintf(format, ap); }
void th_printf(const char *p_fmt, ...) {
  va_list args;
  va_start(args, p_fmt);
  (void)th_vprintf(p_fmt, args); /* ignore return */
  va_end(args);
}

char th_getchar() { return getchar(); }

void th_serialport_initialize(void) { pc.baud(115200); }

void th_timestamp(void) {
  unsigned long microSeconds = 0ul;
  /* USER CODE 2 BEGIN */
  microSeconds = us_ticker_read();
  /* USER CODE 2 END */
  /* This message must NOT be changed. */
  th_printf(EE_MSG_TIMESTAMP, microSeconds);
}

void th_timestamp_initialize(void) {
  /* USER CODE 1 BEGIN */
  // Setting up BOTH perf and energy here
  /* USER CODE 1 END */
  /* This message must NOT be changed. */
  th_printf(EE_MSG_TIMESTAMP_MODE);
  /* Always call the timestamp on initialize so that the open-drain output
     is set to "1" (so that we catch a falling edge) */
  th_timestamp();
}
