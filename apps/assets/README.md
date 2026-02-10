# DataHound Pro Assets Guide

## 📁 Directory Structure

```
apps/assets/
├── images/
│   ├── logo.png                 # Custom company logo (recommended: 200x60px)
│   ├── logo_small.png          # Small logo for headers (recommended: 100x30px)
│   ├── map_placeholder.png     # Map placeholder image (recommended: 600x400px)
│   └── favicon.ico             # Browser favicon (16x16px or 32x32px)
├── style.css                   # Main application styling
└── README.md                   # This file
```

## 🎨 Logo Customization

### Adding Your Custom Logo

1. **Prepare Your Logo Files:**
   - Main logo: `logo.png` (200x60px recommended)
   - Small logo: `logo_small.png` (100x30px recommended)
   - Favicon: `favicon.ico` (16x16px or 32x32px)

2. **File Requirements:**
   - **Format**: PNG (preferred) or JPG
   - **Background**: Transparent PNG recommended
   - **Quality**: High resolution for crisp display
   - **Size**: Keep file sizes under 500KB for performance

3. **Placement:**
   - Copy your logo files to `apps/assets/images/`
   - The application will automatically detect and use them

### Logo Usage in Application

- **Main Logo**: Used in page headers and main navigation
- **Small Logo**: Used in compact spaces and mobile views
- **Favicon**: Displayed in browser tabs and bookmarks

## 🗺️ Map Placeholder Setup

### Adding Map Placeholder Image

1. **Create/Find Image:**
   - Satellite view or aerial photo of your service area
   - Generic map image showing your region
   - Company location photo

2. **Image Specifications:**
   - **Filename**: `map_placeholder.png`
   - **Size**: 600x400 pixels (recommended)
   - **Format**: PNG or JPG
   - **Content**: Should represent your service area

3. **Usage:**
   - Displayed when Google Maps API is not configured
   - Shows in customer profile location tabs
   - Provides visual context for customer locations

## 🌐 Google Maps Integration

### Setting Up Live Maps

1. **Get Google Maps API Key:**
   - Visit [Google Cloud Console](https://console.cloud.google.com/)
   - Enable Maps Static API
   - Generate API key with appropriate restrictions

2. **Configure in Application:**
   - Edit `apps/pages/14_Customer_Profile_Viewer.py`
   - Find the `get_google_maps_static_url()` function
   - Add your API key to the params list:
     ```python
     params.append(f"key=YOUR_API_KEY_HERE")
     ```

3. **Features Enabled:**
   - Live satellite views of customer locations
   - Automatic address-to-coordinates conversion
   - High-resolution aerial imagery

## 🎨 Styling Customization

### Custom CSS

The main styling file is located at `apps/assets/style.css`. You can customize:

- **Colors**: Brand colors, accent colors, backgrounds
- **Typography**: Fonts, sizes, weights
- **Layout**: Spacing, borders, shadows
- **Components**: Button styles, card designs, animations

### Brand Color Variables

Update these CSS variables to match your brand:

```css
:root {
    --primary-color: #your-primary-color;
    --secondary-color: #your-secondary-color;
    --accent-color: #your-accent-color;
    --text-color: #your-text-color;
    --background-color: #your-background-color;
}
```

## 📱 Favicon Setup

### Browser Icon Configuration

1. **Create Favicon:**
   - Use your logo or company symbol
   - 16x16px or 32x32px square format
   - Save as `favicon.ico`

2. **Streamlit Configuration:**
   - Add to `apps/.streamlit/config.toml`:
     ```toml
     [theme]
     favicon = "apps/assets/images/favicon.ico"
     ```

## 🚀 Performance Tips

### Image Optimization

1. **File Sizes:**
   - Logo: < 100KB
   - Map placeholder: < 500KB
   - Favicon: < 10KB

2. **Formats:**
   - PNG: For logos with transparency
   - JPG: For photos and complex images
   - ICO: For favicons

3. **Compression:**
   - Use tools like TinyPNG or ImageOptim
   - Maintain quality while reducing file size

## 🔧 Troubleshooting

### Common Issues

1. **Logo Not Displaying:**
   - Check file path: `apps/assets/images/logo.png`
   - Verify file permissions
   - Ensure correct filename (case-sensitive)

2. **Map Not Loading:**
   - Verify Google Maps API key
   - Check API key restrictions
   - Ensure Maps Static API is enabled

3. **Styling Issues:**
   - Clear browser cache
   - Check CSS syntax in style.css
   - Verify file paths in imports

### File Path Examples

```
# Correct paths from project root:
apps/assets/images/logo.png
apps/assets/images/map_placeholder.png
apps/assets/style.css

# Incorrect paths:
assets/logo.png
images/logo.png
logo.png
```

## 📞 Support

For technical support with asset integration:

1. Check file paths and permissions
2. Verify image formats and sizes
3. Test with different browsers
4. Review console errors in browser developer tools

---

**Note**: After adding or updating assets, restart the Streamlit application to ensure changes are loaded properly.
