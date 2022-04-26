mkdir -p ~/.streamlit/
echo "[general]
email = \"e0685733@gmail.com\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml