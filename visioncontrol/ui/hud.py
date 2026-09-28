import cv2

class HUD:
    def __init__(self):
        # Colors in BGR format
        self.cyan = (255, 255, 0)
        self.violet = (211, 0, 148)
        self.white = (255, 255, 255)
        
    def render(self, frame, fps):
        """Draws the Phase 1 HUD on the frame."""
        height, width, _ = frame.shape
        
        # Top-left: VISIONCONTROL
        cv2.putText(frame, "VISIONCONTROL", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.cyan, 2)
                    
        # Top-right: ● CAMERA LIVE
        cv2.putText(frame, "o CAMERA LIVE", (width - 220, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.violet, 2)
                    
        # Bottom-left: FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (20, height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.white, 2)
                    
        # Center: small text
        text = "Gesture interface initializing..."
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        cv2.putText(frame, text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.cyan, 1)
                    
        return frame
