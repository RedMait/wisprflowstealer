use anyhow::{anyhow, Result};
use arboard::Clipboard;
use dotenvy::dotenv;
use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use rdev::{listen, Event, EventType, Key as RdevKey};
use reqwest::multipart;
use serde_json::Value;
use std::env;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::runtime::Runtime;

static IS_RECORDING: AtomicBool = AtomicBool::new(false);

fn main() -> Result<()> {
    dotenv().ok();
    let api_key = env::var("GROQ_API_KEY").expect("GROQ_API_KEY не найден в .env");

    println!("⚡ Wispr Flow Clone запущен!");
    println!("👉 Зажмите клавишу F10 для записи голоса. Отпустите F10 для автоввода.");

    let rt = Arc::new(Runtime::new()?);
    let audio_buffer = Arc::new(Mutex::new(Vec::<f32>::new()));

    let audio_buf_clone = Arc::clone(&audio_buffer);
    let rt_clone = Arc::clone(&rt);
    let api_key_clone = api_key.clone();

    // Отслеживание нажатий клавиатуры в отдельном потоке
    std::thread::spawn(move || {
        if let Err(error) = listen(move |event| {
            handle_event(event, &audio_buf_clone, &rt_clone, &api_key_clone);
        }) {
            eprintln!("Ошибка клавиатурного хука: {:?}", error);
        }
    });

    // Удержание основного процесса
    loop {
        std::thread::sleep(Duration::from_secs(1));
    }
}

fn handle_event(
    event: Event,
    audio_buffer: &Arc<Mutex<Vec<f32>>>,
    rt: &Arc<Runtime>,
    api_key: &str,
) {
    match event.event_type {
        EventType::KeyPress(RdevKey::F10) => {
            if !IS_RECORDING.swap(true, Ordering::SeqCst) {
                println!("\n🎤 Запись началась...");
                audio_buffer.lock().unwrap().clear();
            }
        }
        EventType::KeyRelease(RdevKey::F10) => {
            if IS_RECORDING.swap(false, Ordering::SeqCst) {
                println!("⏹ Запись остановлена. Обработка ИИ...");
                let audio_data = audio_buffer.lock().unwrap().clone();
                let api_key = api_key.to_string();

                rt.spawn(async move {
                    if let Err(e) = process_and_inject(audio_data, &api_key).await {
                        eprintln!("❌ Ошибка обработки: {:?}", e);
                    }
                });
            }
        }
        _ => {}
    }
}

async fn process_and_inject(_samples: Vec<f32>, api_key: &str) -> Result<()> {
    // 1. Создание минимального WAV файла в RAM
    let wav_bytes = create_dummy_wav()?; 

    let client = reqwest::Client::new();

    // 2. STT - Groq Whisper
    let form = multipart::Form::new()
        .text("model", "whisper-large-v3-turbo")
        .text("language", "ru")
        .part("file", multipart::Part::bytes(wav_bytes).file_name("audio.wav"));

    let res = client
        .post("https://api.groq.com/openai/v1/audio/transcriptions")
        .bearer_auth(api_key)
        .multipart(form)
        .send()
        .await?;

    let json: Value = res.json().await?;
    let raw_text = json["text"].as_str().unwrap_or("").trim();

    if raw_text.is_empty() {
        return Ok(());
    }

    // 3. LLM - Llama 3.1 8B Cleanup
    let payload = serde_json::json!({
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": "You are a speech cleaner. Remove filler words (эээ, ну, типа, um, uh), fix punctuation and capitalization. Output ONLY cleaned text without quotes or explanation."
            },
            {"role": "user", "content": raw_text}
        ]
    });

    let llm_res = client
        .post("https://api.groq.com/openai/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&payload)
        .send()
        .await?;

    let llm_json: Value = llm_res.json().await?;
    let clean_text = llm_json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or(raw_text)
        .trim();

    println!("✨ Текст: {}", clean_text);

    // 4. Буфер обмена + Вставка (Ctrl+V)
    let mut clipboard = Clipboard::new()?;
    clipboard.set_text(clean_text)?;

    tokio::time::sleep(Duration::from_millis(50)).await;

    let mut enigo = Enigo::new(&Settings::default())?;
    enigo.key(Key::Control, Direction::Press)?;
    enigo.key(Key::Unicode('v'), Direction::Click)?;
    enigo.key(Key::Control, Direction::Release)?;

    Ok(())
}

fn create_dummy_wav() -> Result<Vec<u8>> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
    for _ in 0..16000 {
        writer.write_sample(0i16)?;
    }
    writer.finalize()?;
    Ok(cursor.into_inner())
}
